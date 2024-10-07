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


# kernel path: /tmp/torchinductor_sahanp/3i/c3irryp7snv4aivcbeop5vvah5ojav2uy7g6qmheffpxteg6xied.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.mul]
# Source node to ATen node mapping:
#   embeddings => add
#   embeddings_1 => add_1, mul, mul_1, rsqrt, sub, var_mean
#   embeddings_2 => add_2
#   inputs_embeds => embedding
#   position_embeddings => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg0_1, 0), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg1_1), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-07), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
triton_red_fused_add_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.full([XBLOCK, RBLOCK], 128100, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 128100)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 128100")
        tmp6 = tl.load(in_ptr1 + (r1 + (1536*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp7 < 0
        tmp11 = tl.where(tmp10, tmp9, tmp7)
        tl.device_assert(((0 <= tmp11) & (tmp11 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 512")
        tmp13 = tl.load(in_ptr3 + (r1 + (1536*tmp11)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp6 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight, roffset == 0
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp39 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.full([XBLOCK, RBLOCK], 128100, tl.int32)
        tmp20 = tmp0 + tmp19
        tmp21 = tmp0 < 0
        tmp22 = tl.where(tmp21, tmp20, tmp0)
        tl.device_assert(((0 <= tmp22) & (tmp22 < 128100)) | ~(xmask), "index out of bounds: 0 <= tmp22 < 128100")
        tmp24 = tl.load(in_ptr1 + (r1 + (1536*tmp22)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.full([XBLOCK, RBLOCK], 512, tl.int32)
        tmp26 = tmp7 + tmp25
        tmp27 = tmp7 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp7)
        tl.device_assert(((0 <= tmp28) & (tmp28 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp28 < 512")
        tmp30 = tl.load(in_ptr3 + (r1 + (1536*tmp28)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tmp24 + tmp30
        tmp32 = tmp31 - tmp16
        tmp33 = 1536.0
        tmp34 = tmp17 / tmp33
        tmp35 = 1e-07
        tmp36 = tmp34 + tmp35
        tmp37 = libdevice.rsqrt(tmp36)
        tmp38 = tmp32 * tmp37
        tmp40 = tmp38 * tmp39
        tmp42 = tmp40 + tmp41
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp42, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ek/cekhfuxu22uw4hws6auo4vpmmqfcjew22wh4qjeadostrrpqkuho.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1536*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7a/c7a63eevznpqvt7xefh4gx4wdqmtwixo47rg5ssbf7fwyz2wcr43.py
# Topologically Sorted Source Nodes: [scale, truediv], Original ATen: [aten.sqrt, aten.div]
# Source node to ATen node mapping:
#   scale => full_default_1
#   truediv => div
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 8.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_6, %full_default_1), kwargs = {})
triton_poi_fused_div_sqrt_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sqrt_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xk/cxkdtzkwrkyktlb2k6fy3r2uoxegr6geluccje47bdnsie66t6x2.py
# Topologically Sorted Source Nodes: [masked_fill_, rmask, tensor_1, output, output_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
# Source node to ATen node mapping:
#   masked_fill_ => full_default_4, where_1
#   output => where
#   output_1 => amax, div_1, exp, sub_1, sum_1
#   rmask => full_default_2
#   tensor_1 => full_default_3
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 512, 512], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %full_default_3, %view_12), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%full_default_2, %full_default_4, %div_1), kwargs = {})
triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 12288
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


# kernel path: /tmp/torchinductor_sahanp/bd/cbdwg75e74f4sc4pnqoq5w2gv4pr2qvegqoyei7gfqsf4ahvb6n5.py
# Topologically Sorted Source Nodes: [context_layer_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   context_layer_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
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
    x1 = (xindex // 64) % 24
    x2 = (xindex // 1536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (32768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ap/cappu7s6w6nmig7h5ihp6cqox3tba5ew3fqlppdzz3nlbvc4gvuk.py
# Topologically Sorted Source Nodes: [add, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_3
#   hidden_states_1 => add_4, add_5, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_18, %add_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-07), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %arg14_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %arg15_1), kwargs = {})
triton_red_fused_add_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp9 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 1536.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-07
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t6/ct6bjps423xnpya7pcuu5g35n64ab5udh32umh4c3eb23pxr5t4d.py
# Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => add_6, erf, mul_7, mul_8, mul_9
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, 0.5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_20, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_6), kwargs = {})
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 6144
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


# kernel path: /tmp/torchinductor_sahanp/un/cunahhv4fdqlctinymzefzpjjinih2k2nh6wkov4kjj6iagvrlqv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_264, %full_default_101], 1), kwargs = {})
triton_poi_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 4
    x1 = (xindex // 4)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + (1536*x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = 0.0
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/22/c22wls6bbirjaz5rxvzcfwlr6l2eq55lobvytiy4u54qoojwxe5b.py
# Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   start_logits_1 => clone_96
#   start_loss => amax_24, exp_24, sub_73, sum_25
# Graph fragment:
#   %clone_96 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_96, [1], True), kwargs = {})
#   %sub_73 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_96, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_73,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp5 = tl.load(in_ptr1 + (tl.full([XBLOCK, RBLOCK], 0, tl.int32)), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (4*r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp1 >= tmp1
        tmp3 = tl.full([1, 1], 2, tl.int64)
        tmp4 = tmp1 < tmp3
        tmp6 = tmp1 >= tmp3
        tmp7 = tl.full([1, 1], 4, tl.int64)
        tmp8 = tmp1 < tmp7
        tmp9 = 0.0
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = tl.where(tmp4, tmp5, tmp11)
        tmp13 = tmp0 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = triton_helpers.maximum(_tmp15, tmp14)
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp13, rmask)
    tmp15 = triton_helpers.max2(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp17 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp17 - tmp15
        tmp19 = tl_math.exp(tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eo/ceolodsgdchtiilpuxzfzfnpnnyl6kkhadyirmvw755434bs2m25.py
# Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
# Source node to ATen node mapping:
#   end_logits_1 => clone_97
#   end_loss => amax_25, exp_25, sub_75, sum_28
# Graph fragment:
#   %clone_97 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_25 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_97, [1], True), kwargs = {})
#   %sub_75 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_97, %amax_25), kwargs = {})
#   %exp_25 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_75,), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_25, [1], True), kwargs = {})
triton_red_fused__log_softmax_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp6 = tl.load(in_ptr1 + (tl.full([XBLOCK, RBLOCK], 1, tl.int32)), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (4*r0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 2, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp7 = tmp1 >= tmp4
        tmp8 = tl.full([1, 1], 4, tl.int64)
        tmp9 = tmp1 < tmp8
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp7, tmp10, tmp11)
        tmp13 = tl.where(tmp5, tmp6, tmp12)
        tmp14 = tmp0 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = triton_helpers.maximum(_tmp16, tmp15)
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp14, rmask)
    tmp16 = triton_helpers.max2(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp16, None)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp18 = tl.load(out_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp18 - tmp16
        tmp20 = tl_math.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pn/cpnto43y5us5t5uvcukft7u7e6byrnfaaqfh3k2hapmvgy46dnw4.py
# Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_48, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_48 => add_171
#   end_loss => convert_element_type_25, div_49, full_default_100, ne_4, ne_5, neg_1, sum_29, sum_30, where_51
#   end_positions => clamp_max_1, clamp_min_1
#   start_loss => convert_element_type_24, div_48, full_default_98, ne_1, ne_2, neg, sum_26, sum_27, where_49
#   start_positions => clamp_max, clamp_min
#   total_loss => div_50
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg392_1, 0), kwargs = {})
#   %clamp_max : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 512), kwargs = {})
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %full_default_98 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_49 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_98), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_49,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max, 512), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_48 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type_24), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%arg393_1, 0), kwargs = {})
#   %clamp_max_1 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 512), kwargs = {})
#   %ne_4 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %full_default_100 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_51 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %neg_1, %full_default_100), kwargs = {})
#   %sum_30 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_51,), kwargs = {})
#   %ne_5 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%clamp_max_1, 512), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_5,), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_29, torch.float32), kwargs = {})
#   %div_49 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_30, %convert_element_type_25), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_48, %div_49), kwargs = {})
#   %div_50 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_171, 2), kwargs = {})
triton_poi_fused_add_clamp_div_nll_loss_forward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {8: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp14 = tl.load(in_out_ptr0 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp27 = tl.load(in_ptr3 + (0))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp38 = tl.load(in_ptr5 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp41 = tl.load(in_ptr6 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp7 = tl.where(tmp6, tmp5, tmp2)
    tmp8 = tl.full([XBLOCK], 512, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 512), "index out of bounds: 0 <= tmp11 < 512")
    tmp13 = tl.load(in_ptr1 + (tmp11), None, eviction_policy='evict_last')
    tmp16 = tmp13 - tmp15
    tmp19 = tl_math.log(tmp18)
    tmp20 = tmp16 - tmp19
    tmp21 = -tmp20
    tmp22 = 0.0
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tmp6.to(tl.int64)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp29 = triton_helpers.maximum(tmp28, tmp2)
    tmp30 = triton_helpers.minimum(tmp29, tmp4)
    tmp31 = tmp30 != tmp4
    tmp32 = tl.where(tmp31, tmp30, tmp2)
    tmp33 = tmp32 + tmp8
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 512), "index out of bounds: 0 <= tmp35 < 512")
    tmp37 = tl.load(in_ptr4 + (tmp35), None, eviction_policy='evict_last')
    tmp40 = tmp37 - tmp39
    tmp43 = tl_math.log(tmp42)
    tmp44 = tmp40 - tmp43
    tmp45 = -tmp44
    tmp46 = tl.where(tmp31, tmp45, tmp22)
    tmp47 = tmp31.to(tl.int64)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 / tmp48
    tmp50 = tmp26 + tmp49
    tmp51 = 0.5
    tmp52 = tmp50 * tmp51
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp52, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 512), (512, 1))
    assert_size_stride(arg1_1, (1, 512), (512, 1))
    assert_size_stride(arg2_1, (128100, 1536), (1536, 1))
    assert_size_stride(arg3_1, (512, 1536), (1536, 1))
    assert_size_stride(arg4_1, (1536, ), (1, ))
    assert_size_stride(arg5_1, (1536, ), (1, ))
    assert_size_stride(arg6_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg7_1, (1536, ), (1, ))
    assert_size_stride(arg8_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg9_1, (1536, ), (1, ))
    assert_size_stride(arg10_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg11_1, (1536, ), (1, ))
    assert_size_stride(arg12_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg13_1, (1536, ), (1, ))
    assert_size_stride(arg14_1, (1536, ), (1, ))
    assert_size_stride(arg15_1, (1536, ), (1, ))
    assert_size_stride(arg16_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg17_1, (6144, ), (1, ))
    assert_size_stride(arg18_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg19_1, (1536, ), (1, ))
    assert_size_stride(arg20_1, (1536, ), (1, ))
    assert_size_stride(arg21_1, (1536, ), (1, ))
    assert_size_stride(arg22_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg23_1, (1536, ), (1, ))
    assert_size_stride(arg24_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg25_1, (1536, ), (1, ))
    assert_size_stride(arg26_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg27_1, (1536, ), (1, ))
    assert_size_stride(arg28_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg29_1, (1536, ), (1, ))
    assert_size_stride(arg30_1, (1536, ), (1, ))
    assert_size_stride(arg31_1, (1536, ), (1, ))
    assert_size_stride(arg32_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg33_1, (6144, ), (1, ))
    assert_size_stride(arg34_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (1536, ), (1, ))
    assert_size_stride(arg38_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg39_1, (1536, ), (1, ))
    assert_size_stride(arg40_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg41_1, (1536, ), (1, ))
    assert_size_stride(arg42_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg43_1, (1536, ), (1, ))
    assert_size_stride(arg44_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg45_1, (1536, ), (1, ))
    assert_size_stride(arg46_1, (1536, ), (1, ))
    assert_size_stride(arg47_1, (1536, ), (1, ))
    assert_size_stride(arg48_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg49_1, (6144, ), (1, ))
    assert_size_stride(arg50_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg51_1, (1536, ), (1, ))
    assert_size_stride(arg52_1, (1536, ), (1, ))
    assert_size_stride(arg53_1, (1536, ), (1, ))
    assert_size_stride(arg54_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg57_1, (1536, ), (1, ))
    assert_size_stride(arg58_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg61_1, (1536, ), (1, ))
    assert_size_stride(arg62_1, (1536, ), (1, ))
    assert_size_stride(arg63_1, (1536, ), (1, ))
    assert_size_stride(arg64_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg65_1, (6144, ), (1, ))
    assert_size_stride(arg66_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg67_1, (1536, ), (1, ))
    assert_size_stride(arg68_1, (1536, ), (1, ))
    assert_size_stride(arg69_1, (1536, ), (1, ))
    assert_size_stride(arg70_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg73_1, (1536, ), (1, ))
    assert_size_stride(arg74_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg75_1, (1536, ), (1, ))
    assert_size_stride(arg76_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg77_1, (1536, ), (1, ))
    assert_size_stride(arg78_1, (1536, ), (1, ))
    assert_size_stride(arg79_1, (1536, ), (1, ))
    assert_size_stride(arg80_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg81_1, (6144, ), (1, ))
    assert_size_stride(arg82_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg87_1, (1536, ), (1, ))
    assert_size_stride(arg88_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg89_1, (1536, ), (1, ))
    assert_size_stride(arg90_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg91_1, (1536, ), (1, ))
    assert_size_stride(arg92_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg93_1, (1536, ), (1, ))
    assert_size_stride(arg94_1, (1536, ), (1, ))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg97_1, (6144, ), (1, ))
    assert_size_stride(arg98_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg99_1, (1536, ), (1, ))
    assert_size_stride(arg100_1, (1536, ), (1, ))
    assert_size_stride(arg101_1, (1536, ), (1, ))
    assert_size_stride(arg102_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg103_1, (1536, ), (1, ))
    assert_size_stride(arg104_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (1536, ), (1, ))
    assert_size_stride(arg111_1, (1536, ), (1, ))
    assert_size_stride(arg112_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg113_1, (6144, ), (1, ))
    assert_size_stride(arg114_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg119_1, (1536, ), (1, ))
    assert_size_stride(arg120_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg123_1, (1536, ), (1, ))
    assert_size_stride(arg124_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (1536, ), (1, ))
    assert_size_stride(arg127_1, (1536, ), (1, ))
    assert_size_stride(arg128_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg129_1, (6144, ), (1, ))
    assert_size_stride(arg130_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg131_1, (1536, ), (1, ))
    assert_size_stride(arg132_1, (1536, ), (1, ))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg137_1, (1536, ), (1, ))
    assert_size_stride(arg138_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg139_1, (1536, ), (1, ))
    assert_size_stride(arg140_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg141_1, (1536, ), (1, ))
    assert_size_stride(arg142_1, (1536, ), (1, ))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg145_1, (6144, ), (1, ))
    assert_size_stride(arg146_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg147_1, (1536, ), (1, ))
    assert_size_stride(arg148_1, (1536, ), (1, ))
    assert_size_stride(arg149_1, (1536, ), (1, ))
    assert_size_stride(arg150_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg151_1, (1536, ), (1, ))
    assert_size_stride(arg152_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg153_1, (1536, ), (1, ))
    assert_size_stride(arg154_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg157_1, (1536, ), (1, ))
    assert_size_stride(arg158_1, (1536, ), (1, ))
    assert_size_stride(arg159_1, (1536, ), (1, ))
    assert_size_stride(arg160_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg161_1, (6144, ), (1, ))
    assert_size_stride(arg162_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg163_1, (1536, ), (1, ))
    assert_size_stride(arg164_1, (1536, ), (1, ))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg167_1, (1536, ), (1, ))
    assert_size_stride(arg168_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg169_1, (1536, ), (1, ))
    assert_size_stride(arg170_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg171_1, (1536, ), (1, ))
    assert_size_stride(arg172_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg173_1, (1536, ), (1, ))
    assert_size_stride(arg174_1, (1536, ), (1, ))
    assert_size_stride(arg175_1, (1536, ), (1, ))
    assert_size_stride(arg176_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg177_1, (6144, ), (1, ))
    assert_size_stride(arg178_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg179_1, (1536, ), (1, ))
    assert_size_stride(arg180_1, (1536, ), (1, ))
    assert_size_stride(arg181_1, (1536, ), (1, ))
    assert_size_stride(arg182_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg183_1, (1536, ), (1, ))
    assert_size_stride(arg184_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg185_1, (1536, ), (1, ))
    assert_size_stride(arg186_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg187_1, (1536, ), (1, ))
    assert_size_stride(arg188_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg189_1, (1536, ), (1, ))
    assert_size_stride(arg190_1, (1536, ), (1, ))
    assert_size_stride(arg191_1, (1536, ), (1, ))
    assert_size_stride(arg192_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg193_1, (6144, ), (1, ))
    assert_size_stride(arg194_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg195_1, (1536, ), (1, ))
    assert_size_stride(arg196_1, (1536, ), (1, ))
    assert_size_stride(arg197_1, (1536, ), (1, ))
    assert_size_stride(arg198_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg199_1, (1536, ), (1, ))
    assert_size_stride(arg200_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg201_1, (1536, ), (1, ))
    assert_size_stride(arg202_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg203_1, (1536, ), (1, ))
    assert_size_stride(arg204_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (1536, ), (1, ))
    assert_size_stride(arg208_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg209_1, (6144, ), (1, ))
    assert_size_stride(arg210_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg211_1, (1536, ), (1, ))
    assert_size_stride(arg212_1, (1536, ), (1, ))
    assert_size_stride(arg213_1, (1536, ), (1, ))
    assert_size_stride(arg214_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg215_1, (1536, ), (1, ))
    assert_size_stride(arg216_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg219_1, (1536, ), (1, ))
    assert_size_stride(arg220_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg221_1, (1536, ), (1, ))
    assert_size_stride(arg222_1, (1536, ), (1, ))
    assert_size_stride(arg223_1, (1536, ), (1, ))
    assert_size_stride(arg224_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg225_1, (6144, ), (1, ))
    assert_size_stride(arg226_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg227_1, (1536, ), (1, ))
    assert_size_stride(arg228_1, (1536, ), (1, ))
    assert_size_stride(arg229_1, (1536, ), (1, ))
    assert_size_stride(arg230_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg231_1, (1536, ), (1, ))
    assert_size_stride(arg232_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg233_1, (1536, ), (1, ))
    assert_size_stride(arg234_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg235_1, (1536, ), (1, ))
    assert_size_stride(arg236_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg237_1, (1536, ), (1, ))
    assert_size_stride(arg238_1, (1536, ), (1, ))
    assert_size_stride(arg239_1, (1536, ), (1, ))
    assert_size_stride(arg240_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg241_1, (6144, ), (1, ))
    assert_size_stride(arg242_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg243_1, (1536, ), (1, ))
    assert_size_stride(arg244_1, (1536, ), (1, ))
    assert_size_stride(arg245_1, (1536, ), (1, ))
    assert_size_stride(arg246_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg247_1, (1536, ), (1, ))
    assert_size_stride(arg248_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg249_1, (1536, ), (1, ))
    assert_size_stride(arg250_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg251_1, (1536, ), (1, ))
    assert_size_stride(arg252_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg253_1, (1536, ), (1, ))
    assert_size_stride(arg254_1, (1536, ), (1, ))
    assert_size_stride(arg255_1, (1536, ), (1, ))
    assert_size_stride(arg256_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg257_1, (6144, ), (1, ))
    assert_size_stride(arg258_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg259_1, (1536, ), (1, ))
    assert_size_stride(arg260_1, (1536, ), (1, ))
    assert_size_stride(arg261_1, (1536, ), (1, ))
    assert_size_stride(arg262_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg263_1, (1536, ), (1, ))
    assert_size_stride(arg264_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg267_1, (1536, ), (1, ))
    assert_size_stride(arg268_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg269_1, (1536, ), (1, ))
    assert_size_stride(arg270_1, (1536, ), (1, ))
    assert_size_stride(arg271_1, (1536, ), (1, ))
    assert_size_stride(arg272_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg273_1, (6144, ), (1, ))
    assert_size_stride(arg274_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg275_1, (1536, ), (1, ))
    assert_size_stride(arg276_1, (1536, ), (1, ))
    assert_size_stride(arg277_1, (1536, ), (1, ))
    assert_size_stride(arg278_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg279_1, (1536, ), (1, ))
    assert_size_stride(arg280_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg281_1, (1536, ), (1, ))
    assert_size_stride(arg282_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg283_1, (1536, ), (1, ))
    assert_size_stride(arg284_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg285_1, (1536, ), (1, ))
    assert_size_stride(arg286_1, (1536, ), (1, ))
    assert_size_stride(arg287_1, (1536, ), (1, ))
    assert_size_stride(arg288_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg289_1, (6144, ), (1, ))
    assert_size_stride(arg290_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg291_1, (1536, ), (1, ))
    assert_size_stride(arg292_1, (1536, ), (1, ))
    assert_size_stride(arg293_1, (1536, ), (1, ))
    assert_size_stride(arg294_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg295_1, (1536, ), (1, ))
    assert_size_stride(arg296_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg297_1, (1536, ), (1, ))
    assert_size_stride(arg298_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg299_1, (1536, ), (1, ))
    assert_size_stride(arg300_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg301_1, (1536, ), (1, ))
    assert_size_stride(arg302_1, (1536, ), (1, ))
    assert_size_stride(arg303_1, (1536, ), (1, ))
    assert_size_stride(arg304_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg305_1, (6144, ), (1, ))
    assert_size_stride(arg306_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg307_1, (1536, ), (1, ))
    assert_size_stride(arg308_1, (1536, ), (1, ))
    assert_size_stride(arg309_1, (1536, ), (1, ))
    assert_size_stride(arg310_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg311_1, (1536, ), (1, ))
    assert_size_stride(arg312_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg313_1, (1536, ), (1, ))
    assert_size_stride(arg314_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg315_1, (1536, ), (1, ))
    assert_size_stride(arg316_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg317_1, (1536, ), (1, ))
    assert_size_stride(arg318_1, (1536, ), (1, ))
    assert_size_stride(arg319_1, (1536, ), (1, ))
    assert_size_stride(arg320_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg321_1, (6144, ), (1, ))
    assert_size_stride(arg322_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg323_1, (1536, ), (1, ))
    assert_size_stride(arg324_1, (1536, ), (1, ))
    assert_size_stride(arg325_1, (1536, ), (1, ))
    assert_size_stride(arg326_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg327_1, (1536, ), (1, ))
    assert_size_stride(arg328_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg329_1, (1536, ), (1, ))
    assert_size_stride(arg330_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg331_1, (1536, ), (1, ))
    assert_size_stride(arg332_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg333_1, (1536, ), (1, ))
    assert_size_stride(arg334_1, (1536, ), (1, ))
    assert_size_stride(arg335_1, (1536, ), (1, ))
    assert_size_stride(arg336_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg337_1, (6144, ), (1, ))
    assert_size_stride(arg338_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg339_1, (1536, ), (1, ))
    assert_size_stride(arg340_1, (1536, ), (1, ))
    assert_size_stride(arg341_1, (1536, ), (1, ))
    assert_size_stride(arg342_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg343_1, (1536, ), (1, ))
    assert_size_stride(arg344_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg345_1, (1536, ), (1, ))
    assert_size_stride(arg346_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg347_1, (1536, ), (1, ))
    assert_size_stride(arg348_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg349_1, (1536, ), (1, ))
    assert_size_stride(arg350_1, (1536, ), (1, ))
    assert_size_stride(arg351_1, (1536, ), (1, ))
    assert_size_stride(arg352_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg353_1, (6144, ), (1, ))
    assert_size_stride(arg354_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg355_1, (1536, ), (1, ))
    assert_size_stride(arg356_1, (1536, ), (1, ))
    assert_size_stride(arg357_1, (1536, ), (1, ))
    assert_size_stride(arg358_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg359_1, (1536, ), (1, ))
    assert_size_stride(arg360_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg361_1, (1536, ), (1, ))
    assert_size_stride(arg362_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg363_1, (1536, ), (1, ))
    assert_size_stride(arg364_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg365_1, (1536, ), (1, ))
    assert_size_stride(arg366_1, (1536, ), (1, ))
    assert_size_stride(arg367_1, (1536, ), (1, ))
    assert_size_stride(arg368_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg369_1, (6144, ), (1, ))
    assert_size_stride(arg370_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg371_1, (1536, ), (1, ))
    assert_size_stride(arg372_1, (1536, ), (1, ))
    assert_size_stride(arg373_1, (1536, ), (1, ))
    assert_size_stride(arg374_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg375_1, (1536, ), (1, ))
    assert_size_stride(arg376_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg377_1, (1536, ), (1, ))
    assert_size_stride(arg378_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg379_1, (1536, ), (1, ))
    assert_size_stride(arg380_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg381_1, (1536, ), (1, ))
    assert_size_stride(arg382_1, (1536, ), (1, ))
    assert_size_stride(arg383_1, (1536, ), (1, ))
    assert_size_stride(arg384_1, (6144, 1536), (1536, 1))
    assert_size_stride(arg385_1, (6144, ), (1, ))
    assert_size_stride(arg386_1, (1536, 6144), (6144, 1))
    assert_size_stride(arg387_1, (1536, ), (1, ))
    assert_size_stride(arg388_1, (1536, ), (1, ))
    assert_size_stride(arg389_1, (1536, ), (1, ))
    assert_size_stride(arg390_1, (2, 1536), (1536, 1))
    assert_size_stride(arg391_1, (2, ), (1, ))
    assert_size_stride(arg392_1, (1, ), (1, ))
    assert_size_stride(arg393_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 512, 1536), (786432, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mul_native_layer_norm_0.run(arg0_1, arg2_1, arg1_1, arg3_1, arg4_1, arg5_1, buf3, 512, 1536, grid=grid(512), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((512, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg6_1, (1536, 1536), (1, 1536), 0), out=buf4)
        del arg6_1
        buf5 = empty_strided_cuda((512, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg8_1, (1536, 1536), (1, 1536), 0), out=buf5)
        del arg8_1
        buf6 = empty_strided_cuda((1, 24, 512, 64), (786432, 32768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf4, arg7_1, buf6, 786432, grid=grid(786432), stream=stream0)
        del arg7_1
        buf7 = reinterpret_tensor(buf5, (24, 64, 512), (64, 1, 1536), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [scale, truediv], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf7, arg9_1, 786432, grid=grid(786432), stream=stream0)
        del arg9_1
        buf8 = empty_strided_cuda((24, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scale, truediv, attention_scores], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (24, 512, 64), (32768, 64, 1), 0), buf7, out=buf8)
        buf12 = empty_strided_cuda((1, 24, 512, 512), (6291456, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [masked_fill_, rmask, tensor_1, output, output_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf8, buf12, 12288, 512, grid=grid(12288), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (512, 1536), (1536, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg10_1, (1536, 1536), (1, 1536), 0), out=buf11)
        del arg10_1
        buf13 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg11_1, buf13, 786432, grid=grid(786432), stream=stream0)
        del arg11_1
        buf14 = reinterpret_tensor(buf11, (24, 512, 64), (32768, 64, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [context_layer], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf13, (24, 512, 64), (32768, 64, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf13, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [context_layer_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 786432, grid=grid(786432), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (512, 1536), (1536, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg12_1, (1536, 1536), (1, 1536), 0), out=buf16)
        del arg12_1
        buf20 = reinterpret_tensor(buf15, (1, 512, 1536), (786432, 1536, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [add, hidden_states_1], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf16, arg13_1, buf3, arg14_1, arg15_1, buf20, 512, 1536, grid=grid(512), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        buf21 = empty_strided_cuda((512, 6144), (6144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg16_1, (1536, 6144), (1, 1536), 0), out=buf21)
        del arg16_1
        buf22 = reinterpret_tensor(buf21, (1, 512, 6144), (3145728, 6144, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf22, arg17_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg17_1
        buf23 = reinterpret_tensor(buf3, (512, 1536), (1536, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf22, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg18_1, (6144, 1536), (1, 6144), 0), out=buf23)
        del arg18_1
        buf27 = reinterpret_tensor(buf16, (1, 512, 1536), (786432, 1536, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [add_1, hidden_states_5], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf23, arg19_1, buf20, arg20_1, arg21_1, buf27, 512, 1536, grid=grid(512), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg22_1, (1536, 1536), (1, 1536), 0), out=buf28)
        del arg22_1
        buf29 = reinterpret_tensor(buf20, (512, 1536), (1536, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg24_1, (1536, 1536), (1, 1536), 0), out=buf29)
        del arg24_1
        buf30 = reinterpret_tensor(buf4, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf28, arg23_1, buf30, 786432, grid=grid(786432), stream=stream0)
        del arg23_1
        buf31 = reinterpret_tensor(buf29, (24, 64, 512), (64, 1, 1536), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [scale_1, truediv_1], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf31, arg25_1, 786432, grid=grid(786432), stream=stream0)
        del arg25_1
        buf32 = reinterpret_tensor(buf12, (24, 512, 512), (262144, 512, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [scale_1, truediv_1, attention_scores_2], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (24, 512, 64), (32768, 64, 1), 0), buf31, out=buf32)
        buf36 = reinterpret_tensor(buf8, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__1, rmask_1, tensor_3, output_2, output_3], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf32, buf36, 12288, 512, grid=grid(12288), stream=stream0)
        buf35 = reinterpret_tensor(buf31, (512, 1536), (1536, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg26_1, (1536, 1536), (1, 1536), 0), out=buf35)
        del arg26_1
        buf37 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf35, arg27_1, buf37, 786432, grid=grid(786432), stream=stream0)
        del arg27_1
        buf38 = reinterpret_tensor(buf35, (24, 512, 64), (32768, 64, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [context_layer_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf37, (24, 512, 64), (32768, 64, 1), 0), out=buf38)
        buf39 = reinterpret_tensor(buf37, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [context_layer_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf38, buf39, 786432, grid=grid(786432), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (512, 1536), (1536, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg28_1, (1536, 1536), (1, 1536), 0), out=buf40)
        del arg28_1
        buf44 = reinterpret_tensor(buf39, (1, 512, 1536), (786432, 1536, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [add_2, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf40, arg29_1, buf27, arg30_1, arg31_1, buf44, 512, 1536, grid=grid(512), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        buf45 = reinterpret_tensor(buf22, (512, 6144), (6144, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg32_1, (1536, 6144), (1, 1536), 0), out=buf45)
        del arg32_1
        buf46 = reinterpret_tensor(buf45, (1, 512, 6144), (3145728, 6144, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf46, arg33_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg33_1
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg34_1, (6144, 1536), (1, 6144), 0), out=buf47)
        del arg34_1
        buf51 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [add_3, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf47, arg35_1, buf44, arg36_1, arg37_1, buf51, 512, 1536, grid=grid(512), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        buf52 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg38_1, (1536, 1536), (1, 1536), 0), out=buf52)
        del arg38_1
        buf53 = reinterpret_tensor(buf44, (512, 1536), (1536, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg40_1, (1536, 1536), (1, 1536), 0), out=buf53)
        del arg40_1
        buf54 = reinterpret_tensor(buf28, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf52, arg39_1, buf54, 786432, grid=grid(786432), stream=stream0)
        del arg39_1
        buf55 = reinterpret_tensor(buf53, (24, 64, 512), (64, 1, 1536), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [scale_2, truediv_2], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf55, arg41_1, 786432, grid=grid(786432), stream=stream0)
        del arg41_1
        buf56 = reinterpret_tensor(buf36, (24, 512, 512), (262144, 512, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [scale_2, truediv_2, attention_scores_4], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (24, 512, 64), (32768, 64, 1), 0), buf55, out=buf56)
        buf60 = reinterpret_tensor(buf32, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__2, rmask_2, tensor_5, output_4, output_5], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf56, buf60, 12288, 512, grid=grid(12288), stream=stream0)
        buf59 = reinterpret_tensor(buf55, (512, 1536), (1536, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf51, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg42_1, (1536, 1536), (1, 1536), 0), out=buf59)
        del arg42_1
        buf61 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf59, arg43_1, buf61, 786432, grid=grid(786432), stream=stream0)
        del arg43_1
        buf62 = reinterpret_tensor(buf59, (24, 512, 64), (32768, 64, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [context_layer_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf61, (24, 512, 64), (32768, 64, 1), 0), out=buf62)
        buf63 = reinterpret_tensor(buf61, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [context_layer_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf62, buf63, 786432, grid=grid(786432), stream=stream0)
        buf64 = reinterpret_tensor(buf62, (512, 1536), (1536, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf63, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg44_1, (1536, 1536), (1, 1536), 0), out=buf64)
        del arg44_1
        buf68 = reinterpret_tensor(buf63, (1, 512, 1536), (786432, 1536, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [add_4, hidden_states_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf64, arg45_1, buf51, arg46_1, arg47_1, buf68, 512, 1536, grid=grid(512), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf69 = reinterpret_tensor(buf46, (512, 6144), (6144, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg48_1, (1536, 6144), (1, 1536), 0), out=buf69)
        del arg48_1
        buf70 = reinterpret_tensor(buf69, (1, 512, 6144), (3145728, 6144, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf70, arg49_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg49_1
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf70, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg50_1, (6144, 1536), (1, 6144), 0), out=buf71)
        del arg50_1
        buf75 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [add_5, hidden_states_17], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf71, arg51_1, buf68, arg52_1, arg53_1, buf75, 512, 1536, grid=grid(512), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        buf76 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg54_1, (1536, 1536), (1, 1536), 0), out=buf76)
        del arg54_1
        buf77 = reinterpret_tensor(buf68, (512, 1536), (1536, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg56_1, (1536, 1536), (1, 1536), 0), out=buf77)
        del arg56_1
        buf78 = reinterpret_tensor(buf52, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [contiguous_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf76, arg55_1, buf78, 786432, grid=grid(786432), stream=stream0)
        del arg55_1
        buf79 = reinterpret_tensor(buf77, (24, 64, 512), (64, 1, 1536), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [scale_3, truediv_3], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf79, arg57_1, 786432, grid=grid(786432), stream=stream0)
        del arg57_1
        buf80 = reinterpret_tensor(buf60, (24, 512, 512), (262144, 512, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [scale_3, truediv_3, attention_scores_6], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (24, 512, 64), (32768, 64, 1), 0), buf79, out=buf80)
        buf84 = reinterpret_tensor(buf56, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__3, rmask_3, tensor_7, output_6, output_7], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf80, buf84, 12288, 512, grid=grid(12288), stream=stream0)
        buf83 = reinterpret_tensor(buf79, (512, 1536), (1536, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg58_1, (1536, 1536), (1, 1536), 0), out=buf83)
        del arg58_1
        buf85 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf83, arg59_1, buf85, 786432, grid=grid(786432), stream=stream0)
        del arg59_1
        buf86 = reinterpret_tensor(buf83, (24, 512, 64), (32768, 64, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [context_layer_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf85, (24, 512, 64), (32768, 64, 1), 0), out=buf86)
        buf87 = reinterpret_tensor(buf85, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [context_layer_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf86, buf87, 786432, grid=grid(786432), stream=stream0)
        buf88 = reinterpret_tensor(buf86, (512, 1536), (1536, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg60_1, (1536, 1536), (1, 1536), 0), out=buf88)
        del arg60_1
        buf92 = reinterpret_tensor(buf87, (1, 512, 1536), (786432, 1536, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [add_6, hidden_states_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf88, arg61_1, buf75, arg62_1, arg63_1, buf92, 512, 1536, grid=grid(512), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        buf93 = reinterpret_tensor(buf70, (512, 6144), (6144, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg64_1, (1536, 6144), (1, 1536), 0), out=buf93)
        del arg64_1
        buf94 = reinterpret_tensor(buf93, (1, 512, 6144), (3145728, 6144, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf94, arg65_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg65_1
        buf95 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf94, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg66_1, (6144, 1536), (1, 6144), 0), out=buf95)
        del arg66_1
        buf99 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [add_7, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf95, arg67_1, buf92, arg68_1, arg69_1, buf99, 512, 1536, grid=grid(512), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf100 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg70_1, (1536, 1536), (1, 1536), 0), out=buf100)
        del arg70_1
        buf101 = reinterpret_tensor(buf92, (512, 1536), (1536, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg72_1, (1536, 1536), (1, 1536), 0), out=buf101)
        del arg72_1
        buf102 = reinterpret_tensor(buf76, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [contiguous_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf100, arg71_1, buf102, 786432, grid=grid(786432), stream=stream0)
        del arg71_1
        buf103 = reinterpret_tensor(buf101, (24, 64, 512), (64, 1, 1536), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [scale_4, truediv_4], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf103, arg73_1, 786432, grid=grid(786432), stream=stream0)
        del arg73_1
        buf104 = reinterpret_tensor(buf84, (24, 512, 512), (262144, 512, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [scale_4, truediv_4, attention_scores_8], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf102, (24, 512, 64), (32768, 64, 1), 0), buf103, out=buf104)
        buf108 = reinterpret_tensor(buf80, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__4, rmask_4, tensor_9, output_8, output_9], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf104, buf108, 12288, 512, grid=grid(12288), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (512, 1536), (1536, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf99, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg74_1, (1536, 1536), (1, 1536), 0), out=buf107)
        del arg74_1
        buf109 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [contiguous_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf107, arg75_1, buf109, 786432, grid=grid(786432), stream=stream0)
        del arg75_1
        buf110 = reinterpret_tensor(buf107, (24, 512, 64), (32768, 64, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [context_layer_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf109, (24, 512, 64), (32768, 64, 1), 0), out=buf110)
        buf111 = reinterpret_tensor(buf109, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [context_layer_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf110, buf111, 786432, grid=grid(786432), stream=stream0)
        buf112 = reinterpret_tensor(buf110, (512, 1536), (1536, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf111, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg76_1, (1536, 1536), (1, 1536), 0), out=buf112)
        del arg76_1
        buf116 = reinterpret_tensor(buf111, (1, 512, 1536), (786432, 1536, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [add_8, hidden_states_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf112, arg77_1, buf99, arg78_1, arg79_1, buf116, 512, 1536, grid=grid(512), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        buf117 = reinterpret_tensor(buf94, (512, 6144), (6144, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg80_1, (1536, 6144), (1, 1536), 0), out=buf117)
        del arg80_1
        buf118 = reinterpret_tensor(buf117, (1, 512, 6144), (3145728, 6144, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf118, arg81_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg81_1
        buf119 = reinterpret_tensor(buf99, (512, 1536), (1536, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg82_1, (6144, 1536), (1, 6144), 0), out=buf119)
        del arg82_1
        buf123 = reinterpret_tensor(buf112, (1, 512, 1536), (786432, 1536, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf119, arg83_1, buf116, arg84_1, arg85_1, buf123, 512, 1536, grid=grid(512), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        buf124 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg86_1, (1536, 1536), (1, 1536), 0), out=buf124)
        del arg86_1
        buf125 = reinterpret_tensor(buf116, (512, 1536), (1536, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg88_1, (1536, 1536), (1, 1536), 0), out=buf125)
        del arg88_1
        buf126 = reinterpret_tensor(buf100, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf124, arg87_1, buf126, 786432, grid=grid(786432), stream=stream0)
        del arg87_1
        buf127 = reinterpret_tensor(buf125, (24, 64, 512), (64, 1, 1536), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [scale_5, truediv_5], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf127, arg89_1, 786432, grid=grid(786432), stream=stream0)
        del arg89_1
        buf128 = reinterpret_tensor(buf108, (24, 512, 512), (262144, 512, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [scale_5, truediv_5, attention_scores_10], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (24, 512, 64), (32768, 64, 1), 0), buf127, out=buf128)
        buf132 = reinterpret_tensor(buf104, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__5, rmask_5, tensor_11, output_10, output_11], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf128, buf132, 12288, 512, grid=grid(12288), stream=stream0)
        buf131 = reinterpret_tensor(buf127, (512, 1536), (1536, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg90_1, (1536, 1536), (1, 1536), 0), out=buf131)
        del arg90_1
        buf133 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [contiguous_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf131, arg91_1, buf133, 786432, grid=grid(786432), stream=stream0)
        del arg91_1
        buf134 = reinterpret_tensor(buf131, (24, 512, 64), (32768, 64, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [context_layer_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf133, (24, 512, 64), (32768, 64, 1), 0), out=buf134)
        buf135 = reinterpret_tensor(buf133, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [context_layer_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf134, buf135, 786432, grid=grid(786432), stream=stream0)
        buf136 = reinterpret_tensor(buf134, (512, 1536), (1536, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf135, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg92_1, (1536, 1536), (1, 1536), 0), out=buf136)
        del arg92_1
        buf140 = reinterpret_tensor(buf135, (1, 512, 1536), (786432, 1536, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [add_10, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf136, arg93_1, buf123, arg94_1, arg95_1, buf140, 512, 1536, grid=grid(512), stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        buf141 = reinterpret_tensor(buf118, (512, 6144), (6144, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg96_1, (1536, 6144), (1, 1536), 0), out=buf141)
        del arg96_1
        buf142 = reinterpret_tensor(buf141, (1, 512, 6144), (3145728, 6144, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf142, arg97_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg97_1
        buf143 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf142, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg98_1, (6144, 1536), (1, 6144), 0), out=buf143)
        del arg98_1
        buf147 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [add_11, hidden_states_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf143, arg99_1, buf140, arg100_1, arg101_1, buf147, 512, 1536, grid=grid(512), stream=stream0)
        del arg100_1
        del arg101_1
        del arg99_1
        buf148 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg102_1, (1536, 1536), (1, 1536), 0), out=buf148)
        del arg102_1
        buf149 = reinterpret_tensor(buf140, (512, 1536), (1536, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg104_1, (1536, 1536), (1, 1536), 0), out=buf149)
        del arg104_1
        buf150 = reinterpret_tensor(buf124, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [contiguous_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf148, arg103_1, buf150, 786432, grid=grid(786432), stream=stream0)
        del arg103_1
        buf151 = reinterpret_tensor(buf149, (24, 64, 512), (64, 1, 1536), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [scale_6, truediv_6], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf151, arg105_1, 786432, grid=grid(786432), stream=stream0)
        del arg105_1
        buf152 = reinterpret_tensor(buf132, (24, 512, 512), (262144, 512, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [scale_6, truediv_6, attention_scores_12], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (24, 512, 64), (32768, 64, 1), 0), buf151, out=buf152)
        buf156 = reinterpret_tensor(buf128, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__6, rmask_6, tensor_13, output_12, output_13], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf152, buf156, 12288, 512, grid=grid(12288), stream=stream0)
        buf155 = reinterpret_tensor(buf151, (512, 1536), (1536, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg106_1, (1536, 1536), (1, 1536), 0), out=buf155)
        del arg106_1
        buf157 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg107_1, buf157, 786432, grid=grid(786432), stream=stream0)
        del arg107_1
        buf158 = reinterpret_tensor(buf155, (24, 512, 64), (32768, 64, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [context_layer_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf157, (24, 512, 64), (32768, 64, 1), 0), out=buf158)
        buf159 = reinterpret_tensor(buf157, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [context_layer_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf158, buf159, 786432, grid=grid(786432), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (512, 1536), (1536, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg108_1, (1536, 1536), (1, 1536), 0), out=buf160)
        del arg108_1
        buf164 = reinterpret_tensor(buf159, (1, 512, 1536), (786432, 1536, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [add_12, hidden_states_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf160, arg109_1, buf147, arg110_1, arg111_1, buf164, 512, 1536, grid=grid(512), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        buf165 = reinterpret_tensor(buf142, (512, 6144), (6144, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg112_1, (1536, 6144), (1, 1536), 0), out=buf165)
        del arg112_1
        buf166 = reinterpret_tensor(buf165, (1, 512, 6144), (3145728, 6144, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf166, arg113_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg113_1
        buf167 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg114_1, (6144, 1536), (1, 6144), 0), out=buf167)
        del arg114_1
        buf171 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [add_13, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf167, arg115_1, buf164, arg116_1, arg117_1, buf171, 512, 1536, grid=grid(512), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        buf172 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg118_1, (1536, 1536), (1, 1536), 0), out=buf172)
        del arg118_1
        buf173 = reinterpret_tensor(buf164, (512, 1536), (1536, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg120_1, (1536, 1536), (1, 1536), 0), out=buf173)
        del arg120_1
        buf174 = reinterpret_tensor(buf148, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [contiguous_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf172, arg119_1, buf174, 786432, grid=grid(786432), stream=stream0)
        del arg119_1
        buf175 = reinterpret_tensor(buf173, (24, 64, 512), (64, 1, 1536), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [scale_7, truediv_7], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf175, arg121_1, 786432, grid=grid(786432), stream=stream0)
        del arg121_1
        buf176 = reinterpret_tensor(buf156, (24, 512, 512), (262144, 512, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [scale_7, truediv_7, attention_scores_14], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (24, 512, 64), (32768, 64, 1), 0), buf175, out=buf176)
        buf180 = reinterpret_tensor(buf152, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__7, rmask_7, tensor_15, output_14, output_15], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf176, buf180, 12288, 512, grid=grid(12288), stream=stream0)
        buf179 = reinterpret_tensor(buf175, (512, 1536), (1536, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg122_1, (1536, 1536), (1, 1536), 0), out=buf179)
        del arg122_1
        buf181 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [contiguous_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf179, arg123_1, buf181, 786432, grid=grid(786432), stream=stream0)
        del arg123_1
        buf182 = reinterpret_tensor(buf179, (24, 512, 64), (32768, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [context_layer_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf181, (24, 512, 64), (32768, 64, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf181, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [context_layer_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf182, buf183, 786432, grid=grid(786432), stream=stream0)
        buf184 = reinterpret_tensor(buf182, (512, 1536), (1536, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg124_1, (1536, 1536), (1, 1536), 0), out=buf184)
        del arg124_1
        buf188 = reinterpret_tensor(buf183, (1, 512, 1536), (786432, 1536, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [add_14, hidden_states_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf184, arg125_1, buf171, arg126_1, arg127_1, buf188, 512, 1536, grid=grid(512), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        buf189 = reinterpret_tensor(buf166, (512, 6144), (6144, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg128_1, (1536, 6144), (1, 1536), 0), out=buf189)
        del arg128_1
        buf190 = reinterpret_tensor(buf189, (1, 512, 6144), (3145728, 6144, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf190, arg129_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg129_1
        buf191 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg130_1, (6144, 1536), (1, 6144), 0), out=buf191)
        del arg130_1
        buf195 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [add_15, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf191, arg131_1, buf188, arg132_1, arg133_1, buf195, 512, 1536, grid=grid(512), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        buf196 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg134_1, (1536, 1536), (1, 1536), 0), out=buf196)
        del arg134_1
        buf197 = reinterpret_tensor(buf188, (512, 1536), (1536, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg136_1, (1536, 1536), (1, 1536), 0), out=buf197)
        del arg136_1
        buf198 = reinterpret_tensor(buf172, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf196, arg135_1, buf198, 786432, grid=grid(786432), stream=stream0)
        del arg135_1
        buf199 = reinterpret_tensor(buf197, (24, 64, 512), (64, 1, 1536), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [scale_8, truediv_8], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf199, arg137_1, 786432, grid=grid(786432), stream=stream0)
        del arg137_1
        buf200 = reinterpret_tensor(buf180, (24, 512, 512), (262144, 512, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [scale_8, truediv_8, attention_scores_16], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (24, 512, 64), (32768, 64, 1), 0), buf199, out=buf200)
        buf204 = reinterpret_tensor(buf176, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__8, rmask_8, tensor_17, output_16, output_17], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf200, buf204, 12288, 512, grid=grid(12288), stream=stream0)
        buf203 = reinterpret_tensor(buf199, (512, 1536), (1536, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg138_1, (1536, 1536), (1, 1536), 0), out=buf203)
        del arg138_1
        buf205 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [contiguous_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf203, arg139_1, buf205, 786432, grid=grid(786432), stream=stream0)
        del arg139_1
        buf206 = reinterpret_tensor(buf203, (24, 512, 64), (32768, 64, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [context_layer_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf205, (24, 512, 64), (32768, 64, 1), 0), out=buf206)
        buf207 = reinterpret_tensor(buf205, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [context_layer_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf206, buf207, 786432, grid=grid(786432), stream=stream0)
        buf208 = reinterpret_tensor(buf206, (512, 1536), (1536, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg140_1, (1536, 1536), (1, 1536), 0), out=buf208)
        del arg140_1
        buf212 = reinterpret_tensor(buf207, (1, 512, 1536), (786432, 1536, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [add_16, hidden_states_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf208, arg141_1, buf195, arg142_1, arg143_1, buf212, 512, 1536, grid=grid(512), stream=stream0)
        del arg141_1
        del arg142_1
        del arg143_1
        buf213 = reinterpret_tensor(buf190, (512, 6144), (6144, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg144_1, (1536, 6144), (1, 1536), 0), out=buf213)
        del arg144_1
        buf214 = reinterpret_tensor(buf213, (1, 512, 6144), (3145728, 6144, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf214, arg145_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg145_1
        buf215 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf214, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg146_1, (6144, 1536), (1, 6144), 0), out=buf215)
        del arg146_1
        buf219 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [add_17, hidden_states_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf215, arg147_1, buf212, arg148_1, arg149_1, buf219, 512, 1536, grid=grid(512), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        buf220 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg150_1, (1536, 1536), (1, 1536), 0), out=buf220)
        del arg150_1
        buf221 = reinterpret_tensor(buf212, (512, 1536), (1536, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg152_1, (1536, 1536), (1, 1536), 0), out=buf221)
        del arg152_1
        buf222 = reinterpret_tensor(buf196, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [contiguous_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf220, arg151_1, buf222, 786432, grid=grid(786432), stream=stream0)
        del arg151_1
        buf223 = reinterpret_tensor(buf221, (24, 64, 512), (64, 1, 1536), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [scale_9, truediv_9], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf223, arg153_1, 786432, grid=grid(786432), stream=stream0)
        del arg153_1
        buf224 = reinterpret_tensor(buf204, (24, 512, 512), (262144, 512, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [scale_9, truediv_9, attention_scores_18], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf222, (24, 512, 64), (32768, 64, 1), 0), buf223, out=buf224)
        buf228 = reinterpret_tensor(buf200, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__9, rmask_9, tensor_19, output_18, output_19], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf224, buf228, 12288, 512, grid=grid(12288), stream=stream0)
        buf227 = reinterpret_tensor(buf223, (512, 1536), (1536, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg154_1, (1536, 1536), (1, 1536), 0), out=buf227)
        del arg154_1
        buf229 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf227, arg155_1, buf229, 786432, grid=grid(786432), stream=stream0)
        del arg155_1
        buf230 = reinterpret_tensor(buf227, (24, 512, 64), (32768, 64, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [context_layer_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf229, (24, 512, 64), (32768, 64, 1), 0), out=buf230)
        buf231 = reinterpret_tensor(buf229, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [context_layer_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf230, buf231, 786432, grid=grid(786432), stream=stream0)
        buf232 = reinterpret_tensor(buf230, (512, 1536), (1536, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg156_1, (1536, 1536), (1, 1536), 0), out=buf232)
        del arg156_1
        buf236 = reinterpret_tensor(buf231, (1, 512, 1536), (786432, 1536, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [add_18, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf232, arg157_1, buf219, arg158_1, arg159_1, buf236, 512, 1536, grid=grid(512), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        buf237 = reinterpret_tensor(buf214, (512, 6144), (6144, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg160_1, (1536, 6144), (1, 1536), 0), out=buf237)
        del arg160_1
        buf238 = reinterpret_tensor(buf237, (1, 512, 6144), (3145728, 6144, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf238, arg161_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg161_1
        buf239 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg162_1, (6144, 1536), (1, 6144), 0), out=buf239)
        del arg162_1
        buf243 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [add_19, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf239, arg163_1, buf236, arg164_1, arg165_1, buf243, 512, 1536, grid=grid(512), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        buf244 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg166_1, (1536, 1536), (1, 1536), 0), out=buf244)
        del arg166_1
        buf245 = reinterpret_tensor(buf236, (512, 1536), (1536, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg168_1, (1536, 1536), (1, 1536), 0), out=buf245)
        del arg168_1
        buf246 = reinterpret_tensor(buf220, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [contiguous_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf244, arg167_1, buf246, 786432, grid=grid(786432), stream=stream0)
        del arg167_1
        buf247 = reinterpret_tensor(buf245, (24, 64, 512), (64, 1, 1536), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [scale_10, truediv_10], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf247, arg169_1, 786432, grid=grid(786432), stream=stream0)
        del arg169_1
        buf248 = reinterpret_tensor(buf228, (24, 512, 512), (262144, 512, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [scale_10, truediv_10, attention_scores_20], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (24, 512, 64), (32768, 64, 1), 0), buf247, out=buf248)
        buf252 = reinterpret_tensor(buf224, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__10, rmask_10, tensor_21, output_20, output_21], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf248, buf252, 12288, 512, grid=grid(12288), stream=stream0)
        buf251 = reinterpret_tensor(buf247, (512, 1536), (1536, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg170_1, (1536, 1536), (1, 1536), 0), out=buf251)
        del arg170_1
        buf253 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [contiguous_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf251, arg171_1, buf253, 786432, grid=grid(786432), stream=stream0)
        del arg171_1
        buf254 = reinterpret_tensor(buf251, (24, 512, 64), (32768, 64, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [context_layer_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf253, (24, 512, 64), (32768, 64, 1), 0), out=buf254)
        buf255 = reinterpret_tensor(buf253, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [context_layer_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf254, buf255, 786432, grid=grid(786432), stream=stream0)
        buf256 = reinterpret_tensor(buf254, (512, 1536), (1536, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf255, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg172_1, (1536, 1536), (1, 1536), 0), out=buf256)
        del arg172_1
        buf260 = reinterpret_tensor(buf255, (1, 512, 1536), (786432, 1536, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [add_20, hidden_states_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf256, arg173_1, buf243, arg174_1, arg175_1, buf260, 512, 1536, grid=grid(512), stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        buf261 = reinterpret_tensor(buf238, (512, 6144), (6144, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg176_1, (1536, 6144), (1, 1536), 0), out=buf261)
        del arg176_1
        buf262 = reinterpret_tensor(buf261, (1, 512, 6144), (3145728, 6144, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf262, arg177_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg177_1
        buf263 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf262, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg178_1, (6144, 1536), (1, 6144), 0), out=buf263)
        del arg178_1
        buf267 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [add_21, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf263, arg179_1, buf260, arg180_1, arg181_1, buf267, 512, 1536, grid=grid(512), stream=stream0)
        del arg179_1
        del arg180_1
        del arg181_1
        buf268 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg182_1, (1536, 1536), (1, 1536), 0), out=buf268)
        del arg182_1
        buf269 = reinterpret_tensor(buf260, (512, 1536), (1536, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg184_1, (1536, 1536), (1, 1536), 0), out=buf269)
        del arg184_1
        buf270 = reinterpret_tensor(buf244, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf268, arg183_1, buf270, 786432, grid=grid(786432), stream=stream0)
        del arg183_1
        buf271 = reinterpret_tensor(buf269, (24, 64, 512), (64, 1, 1536), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [scale_11, truediv_11], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf271, arg185_1, 786432, grid=grid(786432), stream=stream0)
        del arg185_1
        buf272 = reinterpret_tensor(buf252, (24, 512, 512), (262144, 512, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [scale_11, truediv_11, attention_scores_22], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (24, 512, 64), (32768, 64, 1), 0), buf271, out=buf272)
        buf276 = reinterpret_tensor(buf248, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__11, rmask_11, tensor_23, output_22, output_23], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf272, buf276, 12288, 512, grid=grid(12288), stream=stream0)
        buf275 = reinterpret_tensor(buf271, (512, 1536), (1536, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg186_1, (1536, 1536), (1, 1536), 0), out=buf275)
        del arg186_1
        buf277 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [contiguous_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf275, arg187_1, buf277, 786432, grid=grid(786432), stream=stream0)
        del arg187_1
        buf278 = reinterpret_tensor(buf275, (24, 512, 64), (32768, 64, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [context_layer_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf277, (24, 512, 64), (32768, 64, 1), 0), out=buf278)
        buf279 = reinterpret_tensor(buf277, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [context_layer_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf278, buf279, 786432, grid=grid(786432), stream=stream0)
        buf280 = reinterpret_tensor(buf278, (512, 1536), (1536, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg188_1, (1536, 1536), (1, 1536), 0), out=buf280)
        del arg188_1
        buf284 = reinterpret_tensor(buf279, (1, 512, 1536), (786432, 1536, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [add_22, hidden_states_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf280, arg189_1, buf267, arg190_1, arg191_1, buf284, 512, 1536, grid=grid(512), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        buf285 = reinterpret_tensor(buf262, (512, 6144), (6144, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg192_1, (1536, 6144), (1, 1536), 0), out=buf285)
        del arg192_1
        buf286 = reinterpret_tensor(buf285, (1, 512, 6144), (3145728, 6144, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf286, arg193_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg193_1
        buf287 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg194_1, (6144, 1536), (1, 6144), 0), out=buf287)
        del arg194_1
        buf291 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [add_23, hidden_states_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf287, arg195_1, buf284, arg196_1, arg197_1, buf291, 512, 1536, grid=grid(512), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        buf292 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg198_1, (1536, 1536), (1, 1536), 0), out=buf292)
        del arg198_1
        buf293 = reinterpret_tensor(buf284, (512, 1536), (1536, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg200_1, (1536, 1536), (1, 1536), 0), out=buf293)
        del arg200_1
        buf294 = reinterpret_tensor(buf268, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [contiguous_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf292, arg199_1, buf294, 786432, grid=grid(786432), stream=stream0)
        del arg199_1
        buf295 = reinterpret_tensor(buf293, (24, 64, 512), (64, 1, 1536), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [scale_12, truediv_12], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf295, arg201_1, 786432, grid=grid(786432), stream=stream0)
        del arg201_1
        buf296 = reinterpret_tensor(buf276, (24, 512, 512), (262144, 512, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [scale_12, truediv_12, attention_scores_24], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (24, 512, 64), (32768, 64, 1), 0), buf295, out=buf296)
        buf300 = reinterpret_tensor(buf272, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__12, rmask_12, tensor_25, output_24, output_25], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf296, buf300, 12288, 512, grid=grid(12288), stream=stream0)
        buf299 = reinterpret_tensor(buf295, (512, 1536), (1536, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf291, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg202_1, (1536, 1536), (1, 1536), 0), out=buf299)
        del arg202_1
        buf301 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf299, arg203_1, buf301, 786432, grid=grid(786432), stream=stream0)
        del arg203_1
        buf302 = reinterpret_tensor(buf299, (24, 512, 64), (32768, 64, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [context_layer_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf300, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf301, (24, 512, 64), (32768, 64, 1), 0), out=buf302)
        buf303 = reinterpret_tensor(buf301, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [context_layer_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf302, buf303, 786432, grid=grid(786432), stream=stream0)
        buf304 = reinterpret_tensor(buf302, (512, 1536), (1536, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg204_1, (1536, 1536), (1, 1536), 0), out=buf304)
        del arg204_1
        buf308 = reinterpret_tensor(buf303, (1, 512, 1536), (786432, 1536, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf304, arg205_1, buf291, arg206_1, arg207_1, buf308, 512, 1536, grid=grid(512), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        buf309 = reinterpret_tensor(buf286, (512, 6144), (6144, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg208_1, (1536, 6144), (1, 1536), 0), out=buf309)
        del arg208_1
        buf310 = reinterpret_tensor(buf309, (1, 512, 6144), (3145728, 6144, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf310, arg209_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg209_1
        buf311 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg210_1, (6144, 1536), (1, 6144), 0), out=buf311)
        del arg210_1
        buf315 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [add_25, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf311, arg211_1, buf308, arg212_1, arg213_1, buf315, 512, 1536, grid=grid(512), stream=stream0)
        del arg211_1
        del arg212_1
        del arg213_1
        buf316 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg214_1, (1536, 1536), (1, 1536), 0), out=buf316)
        del arg214_1
        buf317 = reinterpret_tensor(buf308, (512, 1536), (1536, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg216_1, (1536, 1536), (1, 1536), 0), out=buf317)
        del arg216_1
        buf318 = reinterpret_tensor(buf292, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [contiguous_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf316, arg215_1, buf318, 786432, grid=grid(786432), stream=stream0)
        del arg215_1
        buf319 = reinterpret_tensor(buf317, (24, 64, 512), (64, 1, 1536), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [scale_13, truediv_13], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf319, arg217_1, 786432, grid=grid(786432), stream=stream0)
        del arg217_1
        buf320 = reinterpret_tensor(buf300, (24, 512, 512), (262144, 512, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [scale_13, truediv_13, attention_scores_26], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf318, (24, 512, 64), (32768, 64, 1), 0), buf319, out=buf320)
        buf324 = reinterpret_tensor(buf296, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__13, rmask_13, tensor_27, output_26, output_27], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf320, buf324, 12288, 512, grid=grid(12288), stream=stream0)
        buf323 = reinterpret_tensor(buf319, (512, 1536), (1536, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg218_1, (1536, 1536), (1, 1536), 0), out=buf323)
        del arg218_1
        buf325 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [contiguous_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf323, arg219_1, buf325, 786432, grid=grid(786432), stream=stream0)
        del arg219_1
        buf326 = reinterpret_tensor(buf323, (24, 512, 64), (32768, 64, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [context_layer_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf325, (24, 512, 64), (32768, 64, 1), 0), out=buf326)
        buf327 = reinterpret_tensor(buf325, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [context_layer_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf326, buf327, 786432, grid=grid(786432), stream=stream0)
        buf328 = reinterpret_tensor(buf326, (512, 1536), (1536, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg220_1, (1536, 1536), (1, 1536), 0), out=buf328)
        del arg220_1
        buf332 = reinterpret_tensor(buf327, (1, 512, 1536), (786432, 1536, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [add_26, hidden_states_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf328, arg221_1, buf315, arg222_1, arg223_1, buf332, 512, 1536, grid=grid(512), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        buf333 = reinterpret_tensor(buf310, (512, 6144), (6144, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf332, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg224_1, (1536, 6144), (1, 1536), 0), out=buf333)
        del arg224_1
        buf334 = reinterpret_tensor(buf333, (1, 512, 6144), (3145728, 6144, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf334, arg225_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg225_1
        buf335 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg226_1, (6144, 1536), (1, 6144), 0), out=buf335)
        del arg226_1
        buf339 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [add_27, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf335, arg227_1, buf332, arg228_1, arg229_1, buf339, 512, 1536, grid=grid(512), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        buf340 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg230_1, (1536, 1536), (1, 1536), 0), out=buf340)
        del arg230_1
        buf341 = reinterpret_tensor(buf332, (512, 1536), (1536, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg232_1, (1536, 1536), (1, 1536), 0), out=buf341)
        del arg232_1
        buf342 = reinterpret_tensor(buf316, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf340, arg231_1, buf342, 786432, grid=grid(786432), stream=stream0)
        del arg231_1
        buf343 = reinterpret_tensor(buf341, (24, 64, 512), (64, 1, 1536), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [scale_14, truediv_14], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf343, arg233_1, 786432, grid=grid(786432), stream=stream0)
        del arg233_1
        buf344 = reinterpret_tensor(buf324, (24, 512, 512), (262144, 512, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [scale_14, truediv_14, attention_scores_28], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf342, (24, 512, 64), (32768, 64, 1), 0), buf343, out=buf344)
        buf348 = reinterpret_tensor(buf320, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__14, rmask_14, tensor_29, output_28, output_29], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf344, buf348, 12288, 512, grid=grid(12288), stream=stream0)
        buf347 = reinterpret_tensor(buf343, (512, 1536), (1536, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg234_1, (1536, 1536), (1, 1536), 0), out=buf347)
        del arg234_1
        buf349 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [contiguous_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf347, arg235_1, buf349, 786432, grid=grid(786432), stream=stream0)
        del arg235_1
        buf350 = reinterpret_tensor(buf347, (24, 512, 64), (32768, 64, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [context_layer_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf348, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf349, (24, 512, 64), (32768, 64, 1), 0), out=buf350)
        buf351 = reinterpret_tensor(buf349, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [context_layer_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf350, buf351, 786432, grid=grid(786432), stream=stream0)
        buf352 = reinterpret_tensor(buf350, (512, 1536), (1536, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf351, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg236_1, (1536, 1536), (1, 1536), 0), out=buf352)
        del arg236_1
        buf356 = reinterpret_tensor(buf351, (1, 512, 1536), (786432, 1536, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [add_28, hidden_states_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf352, arg237_1, buf339, arg238_1, arg239_1, buf356, 512, 1536, grid=grid(512), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        buf357 = reinterpret_tensor(buf334, (512, 6144), (6144, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg240_1, (1536, 6144), (1, 1536), 0), out=buf357)
        del arg240_1
        buf358 = reinterpret_tensor(buf357, (1, 512, 6144), (3145728, 6144, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf358, arg241_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg241_1
        buf359 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg242_1, (6144, 1536), (1, 6144), 0), out=buf359)
        del arg242_1
        buf363 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_89], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf359, arg243_1, buf356, arg244_1, arg245_1, buf363, 512, 1536, grid=grid(512), stream=stream0)
        del arg243_1
        del arg244_1
        del arg245_1
        buf364 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg246_1, (1536, 1536), (1, 1536), 0), out=buf364)
        del arg246_1
        buf365 = reinterpret_tensor(buf356, (512, 1536), (1536, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg248_1, (1536, 1536), (1, 1536), 0), out=buf365)
        del arg248_1
        buf366 = reinterpret_tensor(buf340, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [contiguous_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf364, arg247_1, buf366, 786432, grid=grid(786432), stream=stream0)
        del arg247_1
        buf367 = reinterpret_tensor(buf365, (24, 64, 512), (64, 1, 1536), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [scale_15, truediv_15], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf367, arg249_1, 786432, grid=grid(786432), stream=stream0)
        del arg249_1
        buf368 = reinterpret_tensor(buf348, (24, 512, 512), (262144, 512, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [scale_15, truediv_15, attention_scores_30], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (24, 512, 64), (32768, 64, 1), 0), buf367, out=buf368)
        buf372 = reinterpret_tensor(buf344, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__15, rmask_15, tensor_31, output_30, output_31], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf368, buf372, 12288, 512, grid=grid(12288), stream=stream0)
        buf371 = reinterpret_tensor(buf367, (512, 1536), (1536, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg250_1, (1536, 1536), (1, 1536), 0), out=buf371)
        del arg250_1
        buf373 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf371, arg251_1, buf373, 786432, grid=grid(786432), stream=stream0)
        del arg251_1
        buf374 = reinterpret_tensor(buf371, (24, 512, 64), (32768, 64, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [context_layer_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf373, (24, 512, 64), (32768, 64, 1), 0), out=buf374)
        buf375 = reinterpret_tensor(buf373, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [context_layer_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf374, buf375, 786432, grid=grid(786432), stream=stream0)
        buf376 = reinterpret_tensor(buf374, (512, 1536), (1536, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg252_1, (1536, 1536), (1, 1536), 0), out=buf376)
        del arg252_1
        buf380 = reinterpret_tensor(buf375, (1, 512, 1536), (786432, 1536, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [add_30, hidden_states_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf376, arg253_1, buf363, arg254_1, arg255_1, buf380, 512, 1536, grid=grid(512), stream=stream0)
        del arg253_1
        del arg254_1
        del arg255_1
        buf381 = reinterpret_tensor(buf358, (512, 6144), (6144, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf380, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg256_1, (1536, 6144), (1, 1536), 0), out=buf381)
        del arg256_1
        buf382 = reinterpret_tensor(buf381, (1, 512, 6144), (3145728, 6144, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf382, arg257_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg257_1
        buf383 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf382, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg258_1, (6144, 1536), (1, 6144), 0), out=buf383)
        del arg258_1
        buf387 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [add_31, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf383, arg259_1, buf380, arg260_1, arg261_1, buf387, 512, 1536, grid=grid(512), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        buf388 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg262_1, (1536, 1536), (1, 1536), 0), out=buf388)
        del arg262_1
        buf389 = reinterpret_tensor(buf380, (512, 1536), (1536, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg264_1, (1536, 1536), (1, 1536), 0), out=buf389)
        del arg264_1
        buf390 = reinterpret_tensor(buf364, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [contiguous_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf388, arg263_1, buf390, 786432, grid=grid(786432), stream=stream0)
        del arg263_1
        buf391 = reinterpret_tensor(buf389, (24, 64, 512), (64, 1, 1536), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [scale_16, truediv_16], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf391, arg265_1, 786432, grid=grid(786432), stream=stream0)
        del arg265_1
        buf392 = reinterpret_tensor(buf372, (24, 512, 512), (262144, 512, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [scale_16, truediv_16, attention_scores_32], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf390, (24, 512, 64), (32768, 64, 1), 0), buf391, out=buf392)
        buf396 = reinterpret_tensor(buf368, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__16, rmask_16, tensor_33, output_32, output_33], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf392, buf396, 12288, 512, grid=grid(12288), stream=stream0)
        buf395 = reinterpret_tensor(buf391, (512, 1536), (1536, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg266_1, (1536, 1536), (1, 1536), 0), out=buf395)
        del arg266_1
        buf397 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [contiguous_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf395, arg267_1, buf397, 786432, grid=grid(786432), stream=stream0)
        del arg267_1
        buf398 = reinterpret_tensor(buf395, (24, 512, 64), (32768, 64, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [context_layer_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf397, (24, 512, 64), (32768, 64, 1), 0), out=buf398)
        buf399 = reinterpret_tensor(buf397, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [context_layer_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf398, buf399, 786432, grid=grid(786432), stream=stream0)
        buf400 = reinterpret_tensor(buf398, (512, 1536), (1536, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg268_1, (1536, 1536), (1, 1536), 0), out=buf400)
        del arg268_1
        buf404 = reinterpret_tensor(buf399, (1, 512, 1536), (786432, 1536, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [add_32, hidden_states_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf400, arg269_1, buf387, arg270_1, arg271_1, buf404, 512, 1536, grid=grid(512), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        buf405 = reinterpret_tensor(buf382, (512, 6144), (6144, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf404, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg272_1, (1536, 6144), (1, 1536), 0), out=buf405)
        del arg272_1
        buf406 = reinterpret_tensor(buf405, (1, 512, 6144), (3145728, 6144, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf406, arg273_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg273_1
        buf407 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg274_1, (6144, 1536), (1, 6144), 0), out=buf407)
        del arg274_1
        buf411 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [add_33, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf407, arg275_1, buf404, arg276_1, arg277_1, buf411, 512, 1536, grid=grid(512), stream=stream0)
        del arg275_1
        del arg276_1
        del arg277_1
        buf412 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg278_1, (1536, 1536), (1, 1536), 0), out=buf412)
        del arg278_1
        buf413 = reinterpret_tensor(buf404, (512, 1536), (1536, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg280_1, (1536, 1536), (1, 1536), 0), out=buf413)
        del arg280_1
        buf414 = reinterpret_tensor(buf388, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf412, arg279_1, buf414, 786432, grid=grid(786432), stream=stream0)
        del arg279_1
        buf415 = reinterpret_tensor(buf413, (24, 64, 512), (64, 1, 1536), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [scale_17, truediv_17], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf415, arg281_1, 786432, grid=grid(786432), stream=stream0)
        del arg281_1
        buf416 = reinterpret_tensor(buf396, (24, 512, 512), (262144, 512, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [scale_17, truediv_17, attention_scores_34], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf414, (24, 512, 64), (32768, 64, 1), 0), buf415, out=buf416)
        buf420 = reinterpret_tensor(buf392, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__17, rmask_17, tensor_35, output_34, output_35], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf416, buf420, 12288, 512, grid=grid(12288), stream=stream0)
        buf419 = reinterpret_tensor(buf415, (512, 1536), (1536, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg282_1, (1536, 1536), (1, 1536), 0), out=buf419)
        del arg282_1
        buf421 = buf414; del buf414  # reuse
        # Topologically Sorted Source Nodes: [contiguous_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf419, arg283_1, buf421, 786432, grid=grid(786432), stream=stream0)
        del arg283_1
        buf422 = reinterpret_tensor(buf419, (24, 512, 64), (32768, 64, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [context_layer_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf420, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf421, (24, 512, 64), (32768, 64, 1), 0), out=buf422)
        buf423 = reinterpret_tensor(buf421, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [context_layer_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf422, buf423, 786432, grid=grid(786432), stream=stream0)
        buf424 = reinterpret_tensor(buf422, (512, 1536), (1536, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg284_1, (1536, 1536), (1, 1536), 0), out=buf424)
        del arg284_1
        buf428 = reinterpret_tensor(buf423, (1, 512, 1536), (786432, 1536, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [add_34, hidden_states_103], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf424, arg285_1, buf411, arg286_1, arg287_1, buf428, 512, 1536, grid=grid(512), stream=stream0)
        del arg285_1
        del arg286_1
        del arg287_1
        buf429 = reinterpret_tensor(buf406, (512, 6144), (6144, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg288_1, (1536, 6144), (1, 1536), 0), out=buf429)
        del arg288_1
        buf430 = reinterpret_tensor(buf429, (1, 512, 6144), (3145728, 6144, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf430, arg289_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg289_1
        buf431 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg290_1, (6144, 1536), (1, 6144), 0), out=buf431)
        del arg290_1
        buf435 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [add_35, hidden_states_107], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf431, arg291_1, buf428, arg292_1, arg293_1, buf435, 512, 1536, grid=grid(512), stream=stream0)
        del arg291_1
        del arg292_1
        del arg293_1
        buf436 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg294_1, (1536, 1536), (1, 1536), 0), out=buf436)
        del arg294_1
        buf437 = reinterpret_tensor(buf428, (512, 1536), (1536, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg296_1, (1536, 1536), (1, 1536), 0), out=buf437)
        del arg296_1
        buf438 = reinterpret_tensor(buf412, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [contiguous_72], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf436, arg295_1, buf438, 786432, grid=grid(786432), stream=stream0)
        del arg295_1
        buf439 = reinterpret_tensor(buf437, (24, 64, 512), (64, 1, 1536), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [scale_18, truediv_18], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf439, arg297_1, 786432, grid=grid(786432), stream=stream0)
        del arg297_1
        buf440 = reinterpret_tensor(buf420, (24, 512, 512), (262144, 512, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [scale_18, truediv_18, attention_scores_36], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (24, 512, 64), (32768, 64, 1), 0), buf439, out=buf440)
        buf444 = reinterpret_tensor(buf416, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__18, rmask_18, tensor_37, output_36, output_37], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf440, buf444, 12288, 512, grid=grid(12288), stream=stream0)
        buf443 = reinterpret_tensor(buf439, (512, 1536), (1536, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf435, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg298_1, (1536, 1536), (1, 1536), 0), out=buf443)
        del arg298_1
        buf445 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [contiguous_74], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf443, arg299_1, buf445, 786432, grid=grid(786432), stream=stream0)
        del arg299_1
        buf446 = reinterpret_tensor(buf443, (24, 512, 64), (32768, 64, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [context_layer_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf444, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf445, (24, 512, 64), (32768, 64, 1), 0), out=buf446)
        buf447 = reinterpret_tensor(buf445, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [context_layer_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf446, buf447, 786432, grid=grid(786432), stream=stream0)
        buf448 = reinterpret_tensor(buf446, (512, 1536), (1536, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf447, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg300_1, (1536, 1536), (1, 1536), 0), out=buf448)
        del arg300_1
        buf452 = reinterpret_tensor(buf447, (1, 512, 1536), (786432, 1536, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [add_36, hidden_states_109], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf448, arg301_1, buf435, arg302_1, arg303_1, buf452, 512, 1536, grid=grid(512), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        buf453 = reinterpret_tensor(buf430, (512, 6144), (6144, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg304_1, (1536, 6144), (1, 1536), 0), out=buf453)
        del arg304_1
        buf454 = reinterpret_tensor(buf453, (1, 512, 6144), (3145728, 6144, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf454, arg305_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg305_1
        buf455 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg306_1, (6144, 1536), (1, 6144), 0), out=buf455)
        del arg306_1
        buf459 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [add_37, hidden_states_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf455, arg307_1, buf452, arg308_1, arg309_1, buf459, 512, 1536, grid=grid(512), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        buf460 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg310_1, (1536, 1536), (1, 1536), 0), out=buf460)
        del arg310_1
        buf461 = reinterpret_tensor(buf452, (512, 1536), (1536, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg312_1, (1536, 1536), (1, 1536), 0), out=buf461)
        del arg312_1
        buf462 = reinterpret_tensor(buf436, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [contiguous_76], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf460, arg311_1, buf462, 786432, grid=grid(786432), stream=stream0)
        del arg311_1
        buf463 = reinterpret_tensor(buf461, (24, 64, 512), (64, 1, 1536), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [scale_19, truediv_19], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf463, arg313_1, 786432, grid=grid(786432), stream=stream0)
        del arg313_1
        buf464 = reinterpret_tensor(buf444, (24, 512, 512), (262144, 512, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [scale_19, truediv_19, attention_scores_38], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (24, 512, 64), (32768, 64, 1), 0), buf463, out=buf464)
        buf468 = reinterpret_tensor(buf440, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__19, rmask_19, tensor_39, output_38, output_39], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf464, buf468, 12288, 512, grid=grid(12288), stream=stream0)
        buf467 = reinterpret_tensor(buf463, (512, 1536), (1536, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg314_1, (1536, 1536), (1, 1536), 0), out=buf467)
        del arg314_1
        buf469 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [contiguous_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf467, arg315_1, buf469, 786432, grid=grid(786432), stream=stream0)
        del arg315_1
        buf470 = reinterpret_tensor(buf467, (24, 512, 64), (32768, 64, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [context_layer_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf468, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf469, (24, 512, 64), (32768, 64, 1), 0), out=buf470)
        buf471 = reinterpret_tensor(buf469, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [context_layer_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf470, buf471, 786432, grid=grid(786432), stream=stream0)
        buf472 = reinterpret_tensor(buf470, (512, 1536), (1536, 1), 0); del buf470  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg316_1, (1536, 1536), (1, 1536), 0), out=buf472)
        del arg316_1
        buf476 = reinterpret_tensor(buf471, (1, 512, 1536), (786432, 1536, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [add_38, hidden_states_115], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf472, arg317_1, buf459, arg318_1, arg319_1, buf476, 512, 1536, grid=grid(512), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        buf477 = reinterpret_tensor(buf454, (512, 6144), (6144, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg320_1, (1536, 6144), (1, 1536), 0), out=buf477)
        del arg320_1
        buf478 = reinterpret_tensor(buf477, (1, 512, 6144), (3145728, 6144, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf478, arg321_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg321_1
        buf479 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg322_1, (6144, 1536), (1, 6144), 0), out=buf479)
        del arg322_1
        buf483 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [add_39, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf479, arg323_1, buf476, arg324_1, arg325_1, buf483, 512, 1536, grid=grid(512), stream=stream0)
        del arg323_1
        del arg324_1
        del arg325_1
        buf484 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg326_1, (1536, 1536), (1, 1536), 0), out=buf484)
        del arg326_1
        buf485 = reinterpret_tensor(buf476, (512, 1536), (1536, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg328_1, (1536, 1536), (1, 1536), 0), out=buf485)
        del arg328_1
        buf486 = reinterpret_tensor(buf460, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [contiguous_80], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf484, arg327_1, buf486, 786432, grid=grid(786432), stream=stream0)
        del arg327_1
        buf487 = reinterpret_tensor(buf485, (24, 64, 512), (64, 1, 1536), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [scale_20, truediv_20], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf487, arg329_1, 786432, grid=grid(786432), stream=stream0)
        del arg329_1
        buf488 = reinterpret_tensor(buf468, (24, 512, 512), (262144, 512, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [scale_20, truediv_20, attention_scores_40], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (24, 512, 64), (32768, 64, 1), 0), buf487, out=buf488)
        buf492 = reinterpret_tensor(buf464, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__20, rmask_20, tensor_41, output_40, output_41], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf488, buf492, 12288, 512, grid=grid(12288), stream=stream0)
        buf491 = reinterpret_tensor(buf487, (512, 1536), (1536, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf483, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg330_1, (1536, 1536), (1, 1536), 0), out=buf491)
        del arg330_1
        buf493 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [contiguous_82], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf491, arg331_1, buf493, 786432, grid=grid(786432), stream=stream0)
        del arg331_1
        buf494 = reinterpret_tensor(buf491, (24, 512, 64), (32768, 64, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [context_layer_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf492, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf493, (24, 512, 64), (32768, 64, 1), 0), out=buf494)
        buf495 = reinterpret_tensor(buf493, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [context_layer_61], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf494, buf495, 786432, grid=grid(786432), stream=stream0)
        buf496 = reinterpret_tensor(buf494, (512, 1536), (1536, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg332_1, (1536, 1536), (1, 1536), 0), out=buf496)
        del arg332_1
        buf500 = reinterpret_tensor(buf495, (1, 512, 1536), (786432, 1536, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [add_40, hidden_states_121], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf496, arg333_1, buf483, arg334_1, arg335_1, buf500, 512, 1536, grid=grid(512), stream=stream0)
        del arg333_1
        del arg334_1
        del arg335_1
        buf501 = reinterpret_tensor(buf478, (512, 6144), (6144, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf500, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg336_1, (1536, 6144), (1, 1536), 0), out=buf501)
        del arg336_1
        buf502 = reinterpret_tensor(buf501, (1, 512, 6144), (3145728, 6144, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf502, arg337_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg337_1
        buf503 = buf496; del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf502, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg338_1, (6144, 1536), (1, 6144), 0), out=buf503)
        del arg338_1
        buf507 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [add_41, hidden_states_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf503, arg339_1, buf500, arg340_1, arg341_1, buf507, 512, 1536, grid=grid(512), stream=stream0)
        del arg339_1
        del arg340_1
        del arg341_1
        buf508 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg342_1, (1536, 1536), (1, 1536), 0), out=buf508)
        del arg342_1
        buf509 = reinterpret_tensor(buf500, (512, 1536), (1536, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg344_1, (1536, 1536), (1, 1536), 0), out=buf509)
        del arg344_1
        buf510 = reinterpret_tensor(buf484, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [contiguous_84], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf508, arg343_1, buf510, 786432, grid=grid(786432), stream=stream0)
        del arg343_1
        buf511 = reinterpret_tensor(buf509, (24, 64, 512), (64, 1, 1536), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [scale_21, truediv_21], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf511, arg345_1, 786432, grid=grid(786432), stream=stream0)
        del arg345_1
        buf512 = reinterpret_tensor(buf492, (24, 512, 512), (262144, 512, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [scale_21, truediv_21, attention_scores_42], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf510, (24, 512, 64), (32768, 64, 1), 0), buf511, out=buf512)
        buf516 = reinterpret_tensor(buf488, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__21, rmask_21, tensor_43, output_42, output_43], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf512, buf516, 12288, 512, grid=grid(12288), stream=stream0)
        buf515 = reinterpret_tensor(buf511, (512, 1536), (1536, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf507, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg346_1, (1536, 1536), (1, 1536), 0), out=buf515)
        del arg346_1
        buf517 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [contiguous_86], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf515, arg347_1, buf517, 786432, grid=grid(786432), stream=stream0)
        del arg347_1
        buf518 = reinterpret_tensor(buf515, (24, 512, 64), (32768, 64, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [context_layer_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf516, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf517, (24, 512, 64), (32768, 64, 1), 0), out=buf518)
        buf519 = reinterpret_tensor(buf517, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [context_layer_64], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf518, buf519, 786432, grid=grid(786432), stream=stream0)
        buf520 = reinterpret_tensor(buf518, (512, 1536), (1536, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf519, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg348_1, (1536, 1536), (1, 1536), 0), out=buf520)
        del arg348_1
        buf524 = reinterpret_tensor(buf519, (1, 512, 1536), (786432, 1536, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [add_42, hidden_states_127], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf520, arg349_1, buf507, arg350_1, arg351_1, buf524, 512, 1536, grid=grid(512), stream=stream0)
        del arg349_1
        del arg350_1
        del arg351_1
        buf525 = reinterpret_tensor(buf502, (512, 6144), (6144, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf524, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg352_1, (1536, 6144), (1, 1536), 0), out=buf525)
        del arg352_1
        buf526 = reinterpret_tensor(buf525, (1, 512, 6144), (3145728, 6144, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf526, arg353_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg353_1
        buf527 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf526, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg354_1, (6144, 1536), (1, 6144), 0), out=buf527)
        del arg354_1
        buf531 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [add_43, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf527, arg355_1, buf524, arg356_1, arg357_1, buf531, 512, 1536, grid=grid(512), stream=stream0)
        del arg355_1
        del arg356_1
        del arg357_1
        buf532 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf531, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg358_1, (1536, 1536), (1, 1536), 0), out=buf532)
        del arg358_1
        buf533 = reinterpret_tensor(buf524, (512, 1536), (1536, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf531, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg360_1, (1536, 1536), (1, 1536), 0), out=buf533)
        del arg360_1
        buf534 = reinterpret_tensor(buf508, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [contiguous_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf532, arg359_1, buf534, 786432, grid=grid(786432), stream=stream0)
        del arg359_1
        buf535 = reinterpret_tensor(buf533, (24, 64, 512), (64, 1, 1536), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [scale_22, truediv_22], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf535, arg361_1, 786432, grid=grid(786432), stream=stream0)
        del arg361_1
        buf536 = reinterpret_tensor(buf516, (24, 512, 512), (262144, 512, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [scale_22, truediv_22, attention_scores_44], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf534, (24, 512, 64), (32768, 64, 1), 0), buf535, out=buf536)
        buf540 = reinterpret_tensor(buf512, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__22, rmask_22, tensor_45, output_44, output_45], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf536, buf540, 12288, 512, grid=grid(12288), stream=stream0)
        buf539 = reinterpret_tensor(buf535, (512, 1536), (1536, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf531, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg362_1, (1536, 1536), (1, 1536), 0), out=buf539)
        del arg362_1
        buf541 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [contiguous_90], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf539, arg363_1, buf541, 786432, grid=grid(786432), stream=stream0)
        del arg363_1
        buf542 = reinterpret_tensor(buf539, (24, 512, 64), (32768, 64, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [context_layer_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf540, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf541, (24, 512, 64), (32768, 64, 1), 0), out=buf542)
        buf543 = reinterpret_tensor(buf541, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [context_layer_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf542, buf543, 786432, grid=grid(786432), stream=stream0)
        buf544 = reinterpret_tensor(buf542, (512, 1536), (1536, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf543, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg364_1, (1536, 1536), (1, 1536), 0), out=buf544)
        del arg364_1
        buf548 = reinterpret_tensor(buf543, (1, 512, 1536), (786432, 1536, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [add_44, hidden_states_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf544, arg365_1, buf531, arg366_1, arg367_1, buf548, 512, 1536, grid=grid(512), stream=stream0)
        del arg365_1
        del arg366_1
        del arg367_1
        buf549 = reinterpret_tensor(buf526, (512, 6144), (6144, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg368_1, (1536, 6144), (1, 1536), 0), out=buf549)
        del arg368_1
        buf550 = reinterpret_tensor(buf549, (1, 512, 6144), (3145728, 6144, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf550, arg369_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg369_1
        buf551 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg370_1, (6144, 1536), (1, 6144), 0), out=buf551)
        del arg370_1
        buf555 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [add_45, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf551, arg371_1, buf548, arg372_1, arg373_1, buf555, 512, 1536, grid=grid(512), stream=stream0)
        del arg371_1
        del arg372_1
        del arg373_1
        buf556 = buf551; del buf551  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg374_1, (1536, 1536), (1, 1536), 0), out=buf556)
        del arg374_1
        buf557 = reinterpret_tensor(buf548, (512, 1536), (1536, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg376_1, (1536, 1536), (1, 1536), 0), out=buf557)
        del arg376_1
        buf558 = reinterpret_tensor(buf532, (1, 24, 512, 64), (786432, 32768, 64, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [contiguous_92], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf556, arg375_1, buf558, 786432, grid=grid(786432), stream=stream0)
        del arg375_1
        del buf556
        buf559 = reinterpret_tensor(buf557, (24, 64, 512), (64, 1, 1536), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [scale_23, truediv_23], Original ATen: [aten.sqrt, aten.div]
        triton_poi_fused_div_sqrt_2.run(buf559, arg377_1, 786432, grid=grid(786432), stream=stream0)
        del arg377_1
        buf560 = reinterpret_tensor(buf540, (24, 512, 512), (262144, 512, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [scale_23, truediv_23, attention_scores_46], Original ATen: [aten.sqrt, aten.div, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf558, (24, 512, 64), (32768, 64, 1), 0), buf559, out=buf560)
        buf564 = reinterpret_tensor(buf536, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [masked_fill__23, rmask_23, tensor_47, output_46, output_47], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.lift_fresh, aten._softmax]
        triton_per_fused__softmax_bitwise_not_lift_fresh_masked_fill_3.run(buf560, buf564, 12288, 512, grid=grid(12288), stream=stream0)
        del buf560
        buf563 = reinterpret_tensor(buf559, (512, 1536), (1536, 1), 0); del buf559  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf555, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg378_1, (1536, 1536), (1, 1536), 0), out=buf563)
        del arg378_1
        buf565 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [contiguous_94], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf563, arg379_1, buf565, 786432, grid=grid(786432), stream=stream0)
        del arg379_1
        buf566 = reinterpret_tensor(buf563, (24, 512, 64), (32768, 64, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [context_layer_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf564, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf565, (24, 512, 64), (32768, 64, 1), 0), out=buf566)
        del buf564
        buf567 = reinterpret_tensor(buf565, (1, 512, 24, 64), (786432, 1536, 64, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [context_layer_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf566, buf567, 786432, grid=grid(786432), stream=stream0)
        buf568 = reinterpret_tensor(buf566, (512, 1536), (1536, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg380_1, (1536, 1536), (1, 1536), 0), out=buf568)
        del arg380_1
        buf572 = reinterpret_tensor(buf567, (1, 512, 1536), (786432, 1536, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [add_46, hidden_states_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf568, arg381_1, buf555, arg382_1, arg383_1, buf572, 512, 1536, grid=grid(512), stream=stream0)
        del arg381_1
        del arg382_1
        del arg383_1
        buf573 = reinterpret_tensor(buf550, (512, 6144), (6144, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (512, 1536), (1536, 1), 0), reinterpret_tensor(arg384_1, (1536, 6144), (1, 1536), 0), out=buf573)
        del arg384_1
        buf574 = reinterpret_tensor(buf573, (1, 512, 6144), (3145728, 6144, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf574, arg385_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg385_1
        buf575 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf574, (512, 6144), (6144, 1), 0), reinterpret_tensor(arg386_1, (6144, 1536), (1, 6144), 0), out=buf575)
        del arg386_1
        del buf574
        buf579 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [add_47, hidden_states_143], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_5.run(buf575, arg387_1, buf572, arg388_1, arg389_1, buf579, 512, 1536, grid=grid(512), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del buf572
        del buf575
        buf580 = empty_strided_cuda((1536, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(arg390_1, buf580, 6144, grid=grid(6144), stream=stream0)
        del arg390_1
        buf581 = empty_strided_cuda((512, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf579, (512, 1536), (1536, 1), 0), buf580, out=buf581)
        del buf579
        del buf580
        buf582 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
        buf583 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf584 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [start_logits_1, start_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_8.run(buf581, arg391_1, buf582, buf583, buf584, 1, 512, grid=grid(1), stream=stream0)
        buf585 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
        buf586 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf587 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [end_logits_1, end_loss], Original ATen: [aten.clone, aten._log_softmax]
        triton_red_fused__log_softmax_clone_9.run(buf581, arg391_1, buf585, buf586, buf587, 1, 512, grid=grid(1), stream=stream0)
        del arg391_1
        del buf581
        buf588 = reinterpret_tensor(buf583, (), (), 0); del buf583  # reuse
        buf589 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [start_positions, start_loss, end_positions, end_loss, add_48, total_loss], Original ATen: [aten.clamp, aten.nll_loss_forward, aten.add, aten.div]
        triton_poi_fused_add_clamp_div_nll_loss_forward_10.run(buf589, arg392_1, buf582, buf584, arg393_1, buf585, buf586, buf587, 1, grid=grid(1), stream=stream0)
        del arg392_1
        del arg393_1
        del buf584
        del buf586
        del buf587
    return (buf589, buf582, buf585, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((128100, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((6144, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((1536, 6144), (6144, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((2, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg393_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('DebertaV2ForQuestionAnswering', benchmark_compiled_module)
