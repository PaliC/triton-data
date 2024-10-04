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
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   hidden_states => add_1
#   hidden_states_2 => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
#   inputs_embeds => mul
#   positions => iota_1
#   positions_1 => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %iota_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg3_1), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg4_1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/fx/cfxy73nuv2pwhebxfzqkdcqusfsoxbnze2dh2urtdlf42xuljqwm.py
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
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 16
    x3 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (131072*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5e/c5eocrcjolf2bagc5co4eibs6cex426ddwst23o4afxtavof35n6.py
# Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_2 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/2n/c2n5qnzlo3bwwoz2b2kxq3g7rv5cwrapu4xuof4ifbmjgjvn6yp2.py
# Topologically Sorted Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weights_3 => amax, div, exp, sub_1, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_15, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_15, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/zw/czwdytaytvgdcse6ylw2wakiidxochal7ixftemzyujhsuesfe6k.py
# Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_3 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/ut/cut2jyjggo6ki3ncke6lnxmlgo4jgx5fa6mdfa2cag4kwiuw3vun.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_4, hidden_states_5], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embedding => embedding
#   hidden_states => add_1
#   hidden_states_4 => add_5
#   hidden_states_5 => add_6, add_7, mul_4, mul_5, rsqrt_1, sub_2, var_mean_1
#   inputs_embeds => mul
#   positions => iota_1
#   positions_1 => embedding_1
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %iota_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_19), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg13_1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg14_1), kwargs = {})
triton_per_fused_add_arange_embedding_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_arange_embedding_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/ia/ciav2ax23jgw4csgenlsfpfaixkzrhule5uy2m2beuz5lv33qhdp.py
# Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_6 => add_8, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_8), kwargs = {})
triton_poi_fused_gelu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/iv/civkujiux35lyf4ol6c227nhsossa2ccnnyusf6ny4w4mbeuyc6b.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_9
#   hidden_states_11 => add_10, add_11, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_23), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_5), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %arg19_1), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg20_1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/o4/co4keteb6lculrzypvf3p676beaiuk47uwgcnmcphfln4d6xwlhd.py
# Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_9
#   hidden_states_13 => add_13
#   hidden_states_14 => add_14, add_15, mul_12, mul_13, rsqrt_3, sub_5, var_mean_3
# Graph fragment:
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_23), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_41), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_13, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_7), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %arg29_1), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg30_1), kwargs = {})
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/k2/ck2bxq5qmf3hrujh7tgrnmz7er5g3bofevtd2ktl5d7udperbz6c.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%permute_132, %full_default_4], 1), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/gp/cgpzjtbeaiuutgws72g3cj74qxfb4psnarpid5h726ddy5bfua2d.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_12, exp_12, sub_37, sum_13
# Graph fragment:
#   %amax_12 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_268, [1], True), kwargs = {})
#   %sub_37 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_268, %amax_12), kwargs = {})
#   %exp_12 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_37,), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_12, [1], True), kwargs = {})
triton_red_fused__log_softmax_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/ev/cevpg5xavqs4wy2nktg6usmngtbdhlcykyvwx4p4r5hwfuq553fr.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div_12, full_default_3, ne_1, ne_2, neg, sum_14, sum_15, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_269, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_269, -100), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_14, torch.float32), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_15, %convert_element_type), kwargs = {})
triton_red_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 128), (128, 1))
    assert_size_stride(arg1_1, (50265, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (1024, ), (1, ))
    assert_size_stride(arg5_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg8_1, (1024, ), (1, ))
    assert_size_stride(arg9_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (1024, ), (1, ))
    assert_size_stride(arg15_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg18_1, (1024, ), (1, ))
    assert_size_stride(arg19_1, (1024, ), (1, ))
    assert_size_stride(arg20_1, (1024, ), (1, ))
    assert_size_stride(arg21_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg28_1, (1024, ), (1, ))
    assert_size_stride(arg29_1, (1024, ), (1, ))
    assert_size_stride(arg30_1, (1024, ), (1, ))
    assert_size_stride(arg31_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, ), (1, ))
    assert_size_stride(arg37_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (1024, ), (1, ))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (1024, ), (1, ))
    assert_size_stride(arg52_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg75_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg90_1, (1024, ), (1, ))
    assert_size_stride(arg91_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg96_1, (4096, ), (1, ))
    assert_size_stride(arg97_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg112_1, (4096, ), (1, ))
    assert_size_stride(arg113_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg128_1, (4096, ), (1, ))
    assert_size_stride(arg129_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (1024, ), (1, ))
    assert_size_stride(arg143_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg144_1, (4096, ), (1, ))
    assert_size_stride(arg145_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (1024, ), (1, ))
    assert_size_stride(arg149_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg160_1, (4096, ), (1, ))
    assert_size_stride(arg161_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg176_1, (4096, ), (1, ))
    assert_size_stride(arg177_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg178_1, (1024, ), (1, ))
    assert_size_stride(arg179_1, (1024, ), (1, ))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg188_1, (1024, ), (1, ))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg192_1, (4096, ), (1, ))
    assert_size_stride(arg193_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg194_1, (1024, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (32, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf3, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg3_1
        del arg4_1
        buf4 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg5_1, (1024, 1024), (1, 1024), 0), out=buf4)
        del arg5_1
        buf5 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 1024), (1, 1024), 0), out=buf5)
        del arg7_1
        buf6 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, arg8_1, buf6, 4194304, grid=grid(4194304), stream=stream0)
        del arg8_1
        buf7 = reinterpret_tensor(buf5, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, arg6_1, buf7, 4194304, grid=grid(4194304), stream=stream0)
        del arg6_1
        buf8 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf6, (512, 64, 128), (8192, 1, 64), 0), out=buf8)
        buf13 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf8, buf13, 65536, 128, grid=grid(65536), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (4096, 1024), (1024, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), out=buf11)
        del arg9_1
        buf12 = reinterpret_tensor(buf3, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg10_1, buf12, 4194304, grid=grid(4194304), stream=stream0)
        del arg10_1
        buf14 = reinterpret_tensor(buf11, (512, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_3, attn_output], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (512, 128, 64), (8192, 64, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf4, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 4194304, grid=grid(4194304), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (4096, 1024), (1024, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg11_1, (1024, 1024), (1, 1024), 0), out=buf16)
        del arg11_1
        buf17 = reinterpret_tensor(buf16, (32, 128, 1024), (131072, 1024, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (32, 128, 1024), (131072, 1024, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_4, hidden_states_5], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        triton_per_fused_add_arange_embedding_mul_native_layer_norm_5.run(buf17, arg0_1, arg1_1, arg2_1, arg12_1, arg13_1, arg14_1, buf21, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg0_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg2_1
        buf22 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg15_1, (1024, 4096), (1, 1024), 0), out=buf22)
        del arg15_1
        buf23 = reinterpret_tensor(buf22, (32, 128, 4096), (524288, 4096, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf23, arg16_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg16_1
        buf24 = reinterpret_tensor(buf21, (4096, 1024), (1024, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 1024), (1, 4096), 0), out=buf24)
        del arg17_1
        buf28 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf17, buf24, arg18_1, arg19_1, arg20_1, buf28, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg19_1
        del arg20_1
        buf29 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg21_1, (1024, 1024), (1, 1024), 0), out=buf29)
        del arg21_1
        buf30 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 1024), (1, 1024), 0), out=buf30)
        del arg23_1
        buf31 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf30, arg24_1, buf31, 4194304, grid=grid(4194304), stream=stream0)
        del arg24_1
        buf32 = reinterpret_tensor(buf30, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, arg22_1, buf32, 4194304, grid=grid(4194304), stream=stream0)
        del arg22_1
        buf33 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf31, (512, 64, 128), (8192, 1, 64), 0), out=buf33)
        buf38 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf33, buf38, 65536, 128, grid=grid(65536), stream=stream0)
        buf36 = reinterpret_tensor(buf32, (4096, 1024), (1024, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg25_1, (1024, 1024), (1, 1024), 0), out=buf36)
        del arg25_1
        buf37 = reinterpret_tensor(buf28, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, arg26_1, buf37, 4194304, grid=grid(4194304), stream=stream0)
        del arg26_1
        buf39 = reinterpret_tensor(buf36, (512, 128, 64), (8192, 64, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7, attn_output_5], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf38, reinterpret_tensor(buf37, (512, 128, 64), (8192, 64, 1), 0), out=buf39)
        buf40 = reinterpret_tensor(buf29, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf39, buf40, 4194304, grid=grid(4194304), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (4096, 1024), (1024, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 1024), (1, 1024), 0), out=buf41)
        del arg27_1
        buf42 = reinterpret_tensor(buf41, (32, 128, 1024), (131072, 1024, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (32, 128, 1024), (131072, 1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf42, buf17, buf24, arg18_1, arg28_1, arg29_1, arg30_1, buf46, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg18_1
        del arg28_1
        del arg29_1
        del arg30_1
        buf47 = reinterpret_tensor(buf23, (4096, 4096), (4096, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg31_1, (1024, 4096), (1, 1024), 0), out=buf47)
        del arg31_1
        buf48 = reinterpret_tensor(buf47, (32, 128, 4096), (524288, 4096, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf48, arg32_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg32_1
        buf49 = reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg33_1, (4096, 1024), (1, 4096), 0), out=buf49)
        del arg33_1
        buf53 = reinterpret_tensor(buf24, (32, 128, 1024), (131072, 1024, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf42, buf49, arg34_1, arg35_1, arg36_1, buf53, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg35_1
        del arg36_1
        buf54 = reinterpret_tensor(buf17, (4096, 1024), (1024, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg37_1, (1024, 1024), (1, 1024), 0), out=buf54)
        del arg37_1
        buf55 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 1024), (1, 1024), 0), out=buf55)
        del arg39_1
        buf56 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf55, arg40_1, buf56, 4194304, grid=grid(4194304), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf55, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf54, arg38_1, buf57, 4194304, grid=grid(4194304), stream=stream0)
        del arg38_1
        buf58 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf56, (512, 64, 128), (8192, 1, 64), 0), out=buf58)
        buf63 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf58, buf63, 65536, 128, grid=grid(65536), stream=stream0)
        buf61 = reinterpret_tensor(buf57, (4096, 1024), (1024, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg41_1, (1024, 1024), (1, 1024), 0), out=buf61)
        del arg41_1
        buf62 = reinterpret_tensor(buf53, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf61, arg42_1, buf62, 4194304, grid=grid(4194304), stream=stream0)
        del arg42_1
        buf64 = reinterpret_tensor(buf61, (512, 128, 64), (8192, 64, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11, attn_output_10], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf63, reinterpret_tensor(buf62, (512, 128, 64), (8192, 64, 1), 0), out=buf64)
        buf65 = reinterpret_tensor(buf54, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf64, buf65, 4194304, grid=grid(4194304), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (4096, 1024), (1024, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 1024), (1, 1024), 0), out=buf66)
        del arg43_1
        buf67 = reinterpret_tensor(buf66, (32, 128, 1024), (131072, 1024, 1), 0); del buf66  # reuse
        buf71 = reinterpret_tensor(buf65, (32, 128, 1024), (131072, 1024, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf67, buf42, buf49, arg34_1, arg44_1, arg45_1, arg46_1, buf71, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg34_1
        del arg44_1
        del arg45_1
        del arg46_1
        buf72 = reinterpret_tensor(buf48, (4096, 4096), (4096, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg47_1, (1024, 4096), (1, 1024), 0), out=buf72)
        del arg47_1
        buf73 = reinterpret_tensor(buf72, (32, 128, 4096), (524288, 4096, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf73, arg48_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg48_1
        buf74 = reinterpret_tensor(buf71, (4096, 1024), (1024, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg49_1, (4096, 1024), (1, 4096), 0), out=buf74)
        del arg49_1
        buf78 = reinterpret_tensor(buf49, (32, 128, 1024), (131072, 1024, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf67, buf74, arg50_1, arg51_1, arg52_1, buf78, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg51_1
        del arg52_1
        buf79 = reinterpret_tensor(buf42, (4096, 1024), (1024, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg53_1, (1024, 1024), (1, 1024), 0), out=buf79)
        del arg53_1
        buf80 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg55_1, (1024, 1024), (1, 1024), 0), out=buf80)
        del arg55_1
        buf81 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf80, arg56_1, buf81, 4194304, grid=grid(4194304), stream=stream0)
        del arg56_1
        buf82 = reinterpret_tensor(buf80, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf79, arg54_1, buf82, 4194304, grid=grid(4194304), stream=stream0)
        del arg54_1
        buf83 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf81, (512, 64, 128), (8192, 1, 64), 0), out=buf83)
        buf88 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf83, buf88, 65536, 128, grid=grid(65536), stream=stream0)
        buf86 = reinterpret_tensor(buf82, (4096, 1024), (1024, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 1024), (1, 1024), 0), out=buf86)
        del arg57_1
        buf87 = reinterpret_tensor(buf78, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf86, arg58_1, buf87, 4194304, grid=grid(4194304), stream=stream0)
        del arg58_1
        buf89 = reinterpret_tensor(buf86, (512, 128, 64), (8192, 64, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15, attn_output_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf88, reinterpret_tensor(buf87, (512, 128, 64), (8192, 64, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf79, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf90, 4194304, grid=grid(4194304), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (4096, 1024), (1024, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg59_1, (1024, 1024), (1, 1024), 0), out=buf91)
        del arg59_1
        buf92 = reinterpret_tensor(buf91, (32, 128, 1024), (131072, 1024, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (32, 128, 1024), (131072, 1024, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf92, buf67, buf74, arg50_1, arg60_1, arg61_1, arg62_1, buf96, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg50_1
        del arg60_1
        del arg61_1
        del arg62_1
        buf97 = reinterpret_tensor(buf73, (4096, 4096), (4096, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg63_1, (1024, 4096), (1, 1024), 0), out=buf97)
        del arg63_1
        buf98 = reinterpret_tensor(buf97, (32, 128, 4096), (524288, 4096, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf98, arg64_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg64_1
        buf99 = reinterpret_tensor(buf96, (4096, 1024), (1024, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg65_1, (4096, 1024), (1, 4096), 0), out=buf99)
        del arg65_1
        buf103 = reinterpret_tensor(buf74, (32, 128, 1024), (131072, 1024, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf92, buf99, arg66_1, arg67_1, arg68_1, buf103, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg67_1
        del arg68_1
        buf104 = reinterpret_tensor(buf67, (4096, 1024), (1024, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 1024), (1, 1024), 0), out=buf104)
        del arg69_1
        buf105 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 1024), (1, 1024), 0), out=buf105)
        del arg71_1
        buf106 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf105, arg72_1, buf106, 4194304, grid=grid(4194304), stream=stream0)
        del arg72_1
        buf107 = reinterpret_tensor(buf105, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, arg70_1, buf107, 4194304, grid=grid(4194304), stream=stream0)
        del arg70_1
        buf108 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf106, (512, 64, 128), (8192, 1, 64), 0), out=buf108)
        buf113 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf108, buf113, 65536, 128, grid=grid(65536), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (4096, 1024), (1024, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 1024), (1, 1024), 0), out=buf111)
        del arg73_1
        buf112 = reinterpret_tensor(buf103, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, arg74_1, buf112, 4194304, grid=grid(4194304), stream=stream0)
        del arg74_1
        buf114 = reinterpret_tensor(buf111, (512, 128, 64), (8192, 64, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19, attn_output_20], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf113, reinterpret_tensor(buf112, (512, 128, 64), (8192, 64, 1), 0), out=buf114)
        buf115 = reinterpret_tensor(buf104, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf114, buf115, 4194304, grid=grid(4194304), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (4096, 1024), (1024, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 1024), (1, 1024), 0), out=buf116)
        del arg75_1
        buf117 = reinterpret_tensor(buf116, (32, 128, 1024), (131072, 1024, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf115, (32, 128, 1024), (131072, 1024, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf117, buf92, buf99, arg66_1, arg76_1, arg77_1, arg78_1, buf121, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg66_1
        del arg76_1
        del arg77_1
        del arg78_1
        buf122 = reinterpret_tensor(buf98, (4096, 4096), (4096, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg79_1, (1024, 4096), (1, 1024), 0), out=buf122)
        del arg79_1
        buf123 = reinterpret_tensor(buf122, (32, 128, 4096), (524288, 4096, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf123, arg80_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg80_1
        buf124 = reinterpret_tensor(buf121, (4096, 1024), (1024, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg81_1, (4096, 1024), (1, 4096), 0), out=buf124)
        del arg81_1
        buf128 = reinterpret_tensor(buf99, (32, 128, 1024), (131072, 1024, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf117, buf124, arg82_1, arg83_1, arg84_1, buf128, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg83_1
        del arg84_1
        buf129 = reinterpret_tensor(buf92, (4096, 1024), (1024, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 1024), (1, 1024), 0), out=buf129)
        del arg85_1
        buf130 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 1024), (1, 1024), 0), out=buf130)
        del arg87_1
        buf131 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf130, arg88_1, buf131, 4194304, grid=grid(4194304), stream=stream0)
        del arg88_1
        buf132 = reinterpret_tensor(buf130, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, arg86_1, buf132, 4194304, grid=grid(4194304), stream=stream0)
        del arg86_1
        buf133 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf131, (512, 64, 128), (8192, 1, 64), 0), out=buf133)
        buf138 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf133, buf138, 65536, 128, grid=grid(65536), stream=stream0)
        buf136 = reinterpret_tensor(buf132, (4096, 1024), (1024, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 1024), (1, 1024), 0), out=buf136)
        del arg89_1
        buf137 = reinterpret_tensor(buf128, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, arg90_1, buf137, 4194304, grid=grid(4194304), stream=stream0)
        del arg90_1
        buf139 = reinterpret_tensor(buf136, (512, 128, 64), (8192, 64, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23, attn_output_25], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf138, reinterpret_tensor(buf137, (512, 128, 64), (8192, 64, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf129, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf139, buf140, 4194304, grid=grid(4194304), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (4096, 1024), (1024, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg91_1, (1024, 1024), (1, 1024), 0), out=buf141)
        del arg91_1
        buf142 = reinterpret_tensor(buf141, (32, 128, 1024), (131072, 1024, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf140, (32, 128, 1024), (131072, 1024, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf142, buf117, buf124, arg82_1, arg92_1, arg93_1, arg94_1, buf146, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg82_1
        del arg92_1
        del arg93_1
        del arg94_1
        buf147 = reinterpret_tensor(buf123, (4096, 4096), (4096, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg95_1, (1024, 4096), (1, 1024), 0), out=buf147)
        del arg95_1
        buf148 = reinterpret_tensor(buf147, (32, 128, 4096), (524288, 4096, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf148, arg96_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg96_1
        buf149 = reinterpret_tensor(buf146, (4096, 1024), (1024, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 1024), (1, 4096), 0), out=buf149)
        del arg97_1
        buf153 = reinterpret_tensor(buf124, (32, 128, 1024), (131072, 1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf142, buf149, arg98_1, arg99_1, arg100_1, buf153, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg99_1
        buf154 = reinterpret_tensor(buf117, (4096, 1024), (1024, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 1024), (1, 1024), 0), out=buf154)
        del arg101_1
        buf155 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 1024), (1, 1024), 0), out=buf155)
        del arg103_1
        buf156 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg104_1, buf156, 4194304, grid=grid(4194304), stream=stream0)
        del arg104_1
        buf157 = reinterpret_tensor(buf155, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf154, arg102_1, buf157, 4194304, grid=grid(4194304), stream=stream0)
        del arg102_1
        buf158 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf156, (512, 64, 128), (8192, 1, 64), 0), out=buf158)
        buf163 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf158, buf163, 65536, 128, grid=grid(65536), stream=stream0)
        buf161 = reinterpret_tensor(buf157, (4096, 1024), (1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 1024), (1, 1024), 0), out=buf161)
        del arg105_1
        buf162 = reinterpret_tensor(buf153, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf161, arg106_1, buf162, 4194304, grid=grid(4194304), stream=stream0)
        del arg106_1
        buf164 = reinterpret_tensor(buf161, (512, 128, 64), (8192, 64, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27, attn_output_30], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf163, reinterpret_tensor(buf162, (512, 128, 64), (8192, 64, 1), 0), out=buf164)
        buf165 = reinterpret_tensor(buf154, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf164, buf165, 4194304, grid=grid(4194304), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (4096, 1024), (1024, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg107_1, (1024, 1024), (1, 1024), 0), out=buf166)
        del arg107_1
        buf167 = reinterpret_tensor(buf166, (32, 128, 1024), (131072, 1024, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf165, (32, 128, 1024), (131072, 1024, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf167, buf142, buf149, arg98_1, arg108_1, arg109_1, arg110_1, buf171, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del arg98_1
        buf172 = reinterpret_tensor(buf148, (4096, 4096), (4096, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 4096), (1, 1024), 0), out=buf172)
        del arg111_1
        buf173 = reinterpret_tensor(buf172, (32, 128, 4096), (524288, 4096, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf173, arg112_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg112_1
        buf174 = reinterpret_tensor(buf171, (4096, 1024), (1024, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 1024), (1, 4096), 0), out=buf174)
        del arg113_1
        buf178 = reinterpret_tensor(buf149, (32, 128, 1024), (131072, 1024, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf167, buf174, arg114_1, arg115_1, arg116_1, buf178, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg115_1
        del arg116_1
        buf179 = reinterpret_tensor(buf142, (4096, 1024), (1024, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1024), (1, 1024), 0), out=buf179)
        del arg117_1
        buf180 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg119_1, (1024, 1024), (1, 1024), 0), out=buf180)
        del arg119_1
        buf181 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, arg120_1, buf181, 4194304, grid=grid(4194304), stream=stream0)
        del arg120_1
        buf182 = reinterpret_tensor(buf180, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, arg118_1, buf182, 4194304, grid=grid(4194304), stream=stream0)
        del arg118_1
        buf183 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf181, (512, 64, 128), (8192, 1, 64), 0), out=buf183)
        buf188 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf183, buf188, 65536, 128, grid=grid(65536), stream=stream0)
        buf186 = reinterpret_tensor(buf182, (4096, 1024), (1024, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg121_1, (1024, 1024), (1, 1024), 0), out=buf186)
        del arg121_1
        buf187 = reinterpret_tensor(buf178, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, arg122_1, buf187, 4194304, grid=grid(4194304), stream=stream0)
        del arg122_1
        buf189 = reinterpret_tensor(buf186, (512, 128, 64), (8192, 64, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31, attn_output_35], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf188, reinterpret_tensor(buf187, (512, 128, 64), (8192, 64, 1), 0), out=buf189)
        buf190 = reinterpret_tensor(buf179, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf189, buf190, 4194304, grid=grid(4194304), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (4096, 1024), (1024, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1024), (1, 1024), 0), out=buf191)
        del arg123_1
        buf192 = reinterpret_tensor(buf191, (32, 128, 1024), (131072, 1024, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf190, (32, 128, 1024), (131072, 1024, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf192, buf167, buf174, arg114_1, arg124_1, arg125_1, arg126_1, buf196, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg114_1
        del arg124_1
        del arg125_1
        del arg126_1
        buf197 = reinterpret_tensor(buf173, (4096, 4096), (4096, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg127_1, (1024, 4096), (1, 1024), 0), out=buf197)
        del arg127_1
        buf198 = reinterpret_tensor(buf197, (32, 128, 4096), (524288, 4096, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf198, arg128_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg128_1
        buf199 = reinterpret_tensor(buf196, (4096, 1024), (1024, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 1024), (1, 4096), 0), out=buf199)
        del arg129_1
        buf203 = reinterpret_tensor(buf174, (32, 128, 1024), (131072, 1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf192, buf199, arg130_1, arg131_1, arg132_1, buf203, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg131_1
        del arg132_1
        buf204 = reinterpret_tensor(buf167, (4096, 1024), (1024, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 1024), (1, 1024), 0), out=buf204)
        del arg133_1
        buf205 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg135_1, (1024, 1024), (1, 1024), 0), out=buf205)
        del arg135_1
        buf206 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf205, arg136_1, buf206, 4194304, grid=grid(4194304), stream=stream0)
        del arg136_1
        buf207 = reinterpret_tensor(buf205, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf204, arg134_1, buf207, 4194304, grid=grid(4194304), stream=stream0)
        del arg134_1
        buf208 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf206, (512, 64, 128), (8192, 1, 64), 0), out=buf208)
        buf213 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf208, buf213, 65536, 128, grid=grid(65536), stream=stream0)
        buf211 = reinterpret_tensor(buf207, (4096, 1024), (1024, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 1024), (1, 1024), 0), out=buf211)
        del arg137_1
        buf212 = reinterpret_tensor(buf203, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, arg138_1, buf212, 4194304, grid=grid(4194304), stream=stream0)
        del arg138_1
        buf214 = reinterpret_tensor(buf211, (512, 128, 64), (8192, 64, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_35, attn_output_40], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf213, reinterpret_tensor(buf212, (512, 128, 64), (8192, 64, 1), 0), out=buf214)
        buf215 = reinterpret_tensor(buf204, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf214, buf215, 4194304, grid=grid(4194304), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (4096, 1024), (1024, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 1024), (1, 1024), 0), out=buf216)
        del arg139_1
        buf217 = reinterpret_tensor(buf216, (32, 128, 1024), (131072, 1024, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf215, (32, 128, 1024), (131072, 1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf217, buf192, buf199, arg130_1, arg140_1, arg141_1, arg142_1, buf221, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg130_1
        del arg140_1
        del arg141_1
        del arg142_1
        buf222 = reinterpret_tensor(buf198, (4096, 4096), (4096, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 4096), (1, 1024), 0), out=buf222)
        del arg143_1
        buf223 = reinterpret_tensor(buf222, (32, 128, 4096), (524288, 4096, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf223, arg144_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg144_1
        buf224 = reinterpret_tensor(buf221, (4096, 1024), (1024, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg145_1, (4096, 1024), (1, 4096), 0), out=buf224)
        del arg145_1
        buf228 = reinterpret_tensor(buf199, (32, 128, 1024), (131072, 1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf217, buf224, arg146_1, arg147_1, arg148_1, buf228, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg147_1
        del arg148_1
        buf229 = reinterpret_tensor(buf192, (4096, 1024), (1024, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 1024), (1, 1024), 0), out=buf229)
        del arg149_1
        buf230 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg151_1, (1024, 1024), (1, 1024), 0), out=buf230)
        del arg151_1
        buf231 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, arg152_1, buf231, 4194304, grid=grid(4194304), stream=stream0)
        del arg152_1
        buf232 = reinterpret_tensor(buf230, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf229, arg150_1, buf232, 4194304, grid=grid(4194304), stream=stream0)
        del arg150_1
        buf233 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf231, (512, 64, 128), (8192, 1, 64), 0), out=buf233)
        buf238 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf233, buf238, 65536, 128, grid=grid(65536), stream=stream0)
        buf236 = reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg153_1, (1024, 1024), (1, 1024), 0), out=buf236)
        del arg153_1
        buf237 = reinterpret_tensor(buf228, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf236, arg154_1, buf237, 4194304, grid=grid(4194304), stream=stream0)
        del arg154_1
        buf239 = reinterpret_tensor(buf236, (512, 128, 64), (8192, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39, attn_output_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf238, reinterpret_tensor(buf237, (512, 128, 64), (8192, 64, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf229, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf239, buf240, 4194304, grid=grid(4194304), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (4096, 1024), (1024, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 1024), (1, 1024), 0), out=buf241)
        del arg155_1
        buf242 = reinterpret_tensor(buf241, (32, 128, 1024), (131072, 1024, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf240, (32, 128, 1024), (131072, 1024, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf242, buf217, buf224, arg146_1, arg156_1, arg157_1, arg158_1, buf246, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg146_1
        del arg156_1
        del arg157_1
        del arg158_1
        buf247 = reinterpret_tensor(buf223, (4096, 4096), (4096, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 4096), (1, 1024), 0), out=buf247)
        del arg159_1
        buf248 = reinterpret_tensor(buf247, (32, 128, 4096), (524288, 4096, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf248, arg160_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg160_1
        buf249 = reinterpret_tensor(buf246, (4096, 1024), (1024, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg161_1, (4096, 1024), (1, 4096), 0), out=buf249)
        del arg161_1
        buf253 = reinterpret_tensor(buf224, (32, 128, 1024), (131072, 1024, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf242, buf249, arg162_1, arg163_1, arg164_1, buf253, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg163_1
        del arg164_1
        buf254 = reinterpret_tensor(buf217, (4096, 1024), (1024, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg165_1, (1024, 1024), (1, 1024), 0), out=buf254)
        del arg165_1
        buf255 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 1024), (1, 1024), 0), out=buf255)
        del arg167_1
        buf256 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf255, arg168_1, buf256, 4194304, grid=grid(4194304), stream=stream0)
        del arg168_1
        buf257 = reinterpret_tensor(buf255, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf254, arg166_1, buf257, 4194304, grid=grid(4194304), stream=stream0)
        del arg166_1
        buf258 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf256, (512, 64, 128), (8192, 1, 64), 0), out=buf258)
        buf263 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf258, buf263, 65536, 128, grid=grid(65536), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (4096, 1024), (1024, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg169_1, (1024, 1024), (1, 1024), 0), out=buf261)
        del arg169_1
        buf262 = reinterpret_tensor(buf253, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf261, arg170_1, buf262, 4194304, grid=grid(4194304), stream=stream0)
        del arg170_1
        buf264 = reinterpret_tensor(buf261, (512, 128, 64), (8192, 64, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43, attn_output_50], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf263, reinterpret_tensor(buf262, (512, 128, 64), (8192, 64, 1), 0), out=buf264)
        buf265 = reinterpret_tensor(buf254, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf264, buf265, 4194304, grid=grid(4194304), stream=stream0)
        buf266 = reinterpret_tensor(buf264, (4096, 1024), (1024, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 1024), (1, 1024), 0), out=buf266)
        del arg171_1
        buf267 = reinterpret_tensor(buf266, (32, 128, 1024), (131072, 1024, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf265, (32, 128, 1024), (131072, 1024, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf267, buf242, buf249, arg162_1, arg172_1, arg173_1, arg174_1, buf271, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg162_1
        del arg172_1
        del arg173_1
        del arg174_1
        buf272 = reinterpret_tensor(buf248, (4096, 4096), (4096, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg175_1, (1024, 4096), (1, 1024), 0), out=buf272)
        del arg175_1
        buf273 = reinterpret_tensor(buf272, (32, 128, 4096), (524288, 4096, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf273, arg176_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg176_1
        buf274 = reinterpret_tensor(buf271, (4096, 1024), (1024, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 1024), (1, 4096), 0), out=buf274)
        del arg177_1
        buf278 = reinterpret_tensor(buf249, (32, 128, 1024), (131072, 1024, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf267, buf274, arg178_1, arg179_1, arg180_1, buf278, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg179_1
        del arg180_1
        buf279 = reinterpret_tensor(buf242, (4096, 1024), (1024, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 1024), (1, 1024), 0), out=buf279)
        del arg181_1
        buf280 = empty_strided_cuda((4096, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 1024), (1, 1024), 0), out=buf280)
        del arg183_1
        buf281 = empty_strided_cuda((32, 16, 128, 64), (131072, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf280, arg184_1, buf281, 4194304, grid=grid(4194304), stream=stream0)
        del arg184_1
        buf282 = reinterpret_tensor(buf280, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf279, arg182_1, buf282, 4194304, grid=grid(4194304), stream=stream0)
        del arg182_1
        buf283 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (512, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf281, (512, 64, 128), (8192, 1, 64), 0), out=buf283)
        buf288 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf283, buf288, 65536, 128, grid=grid(65536), stream=stream0)
        del buf283
        buf286 = reinterpret_tensor(buf282, (4096, 1024), (1024, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg185_1, (1024, 1024), (1, 1024), 0), out=buf286)
        del arg185_1
        buf287 = reinterpret_tensor(buf278, (32, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, arg186_1, buf287, 4194304, grid=grid(4194304), stream=stream0)
        del arg186_1
        buf289 = reinterpret_tensor(buf286, (512, 128, 64), (8192, 64, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_47, attn_output_55], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf288, reinterpret_tensor(buf287, (512, 128, 64), (8192, 64, 1), 0), out=buf289)
        del buf288
        buf290 = reinterpret_tensor(buf279, (32, 128, 16, 64), (131072, 1024, 64, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf289, buf290, 4194304, grid=grid(4194304), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (4096, 1024), (1024, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg187_1, (1024, 1024), (1, 1024), 0), out=buf291)
        del arg187_1
        buf292 = reinterpret_tensor(buf291, (32, 128, 1024), (131072, 1024, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (32, 128, 1024), (131072, 1024, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_103, hidden_states_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf292, buf267, buf274, arg178_1, arg188_1, arg189_1, arg190_1, buf296, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg178_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf267
        buf297 = reinterpret_tensor(buf273, (4096, 4096), (4096, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg191_1, (1024, 4096), (1, 1024), 0), out=buf297)
        del arg191_1
        buf298 = reinterpret_tensor(buf297, (32, 128, 4096), (524288, 4096, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf298, arg192_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg192_1
        buf299 = reinterpret_tensor(buf296, (4096, 1024), (1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg193_1, (4096, 1024), (1, 4096), 0), out=buf299)
        del arg193_1
        del buf298
        buf303 = reinterpret_tensor(buf274, (32, 128, 1024), (131072, 1024, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf292, buf299, arg194_1, arg195_1, arg196_1, buf303, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del buf292
        del buf299
        buf304 = empty_strided_cuda((1024, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(arg1_1, buf304, 51474432, grid=grid(51474432), stream=stream0)
        del arg1_1
        buf305 = empty_strided_cuda((4096, 50268), (50272, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (4096, 1024), (1024, 1), 0), buf304, out=buf305)
        del buf303
        del buf304
        buf306 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf307 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_10.run(buf305, buf306, buf307, 4096, 50265, grid=grid(4096), stream=stream0)
        buf308 = empty_strided_cuda((), (), torch.float32)
        buf310 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_11.run(buf310, arg197_1, buf305, buf306, buf307, 1, 4096, grid=grid(1), stream=stream0)
        del arg197_1
        del buf306
        del buf307
    return (buf310, reinterpret_tensor(buf305, (32, 128, 50265), (6434816, 50272, 1), 0), buf6, buf12, buf31, buf37, buf56, buf62, buf81, buf87, buf106, buf112, buf131, buf137, buf156, buf162, buf181, buf187, buf206, buf212, buf231, buf237, buf256, buf262, buf281, buf287, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50265, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((32, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PegasusForCausalLM', benchmark_compiled_module)
