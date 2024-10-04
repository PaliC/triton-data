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


# kernel path: /tmp/torchinductor_sahanp/wc/cwcpcx3unufqdw26axwidwd7woped4tkvdkxel24qdfvu2bhtxje.py
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
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = xindex % 128
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr2 + (r2 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8008, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 8008)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 8008")
        tmp6 = tl.load(in_ptr1 + (r2 + (2560*tmp4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp10 = tmp8 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
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
        tmp23 = tl.load(in_ptr2 + (r2 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full([XBLOCK, RBLOCK], 8008, tl.int32)
        tmp16 = tmp0 + tmp15
        tmp17 = tmp0 < 0
        tmp18 = tl.where(tmp17, tmp16, tmp0)
        tl.device_assert(((0 <= tmp18) & (tmp18 < 8008)) | ~(xmask), "index out of bounds: 0 <= tmp18 < 8008")
        tmp20 = tl.load(in_ptr1 + (r2 + (2560*tmp18)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = 1.0
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24 - tmp12
        tmp26 = 2560.0
        tmp27 = tmp13 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp31 * tmp32
        tmp35 = tmp33 + tmp34
        tl.store(out_ptr2 + (r2 + (2560*x3)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mv/cmvi7z5gnzurhwipiyk3h6mvhj5ztvykjhlhffgn5x46tonkvjwd.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 80
    x1 = (xindex // 80) % 128
    x2 = (xindex // 10240) % 32
    x3 = (xindex // 327680)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*x2) + (2560*x1) + (327680*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/eh/cehzo3fdkumnooc6adj6r2sqfagzzxt6glgnnea7hutk43vcj3t6.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 80
    x1 = (xindex // 80) % 128
    x2 = (xindex // 10240) % 32
    x3 = (xindex // 327680)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*x2) + (2560*x1) + (327680*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.11180339887498948
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yw/cywdkeyej6cpkxa34fmvkj7edp7lgjpapk3bh554b3j6e2gq2gls.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: /tmp/torchinductor_sahanp/r7/cr75zrsogupeakwgiahjo7nczw22xfje5ius376ikdjuuwvirynp.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 80
    x1 = (xindex // 80) % 32
    x2 = (xindex // 2560) % 128
    x3 = (xindex // 327680)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*x2) + (10240*x1) + (327680*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uj/cujj5pm2pqsmxrpuhbwwvgia273jcib35mgnqx7cfzxrcxyskjf2.py
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
triton_red_fused_add_arange_embedding_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = xindex % 128
    tmp16_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr2 + (r2 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_out_ptr0 + (r2 + (2560*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8008, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 8008)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 8008")
        tmp6 = tl.load(in_ptr1 + (r2 + (2560*tmp4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp10 = tmp8 + tmp9
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_reduce(
            tmp15, tmp16_mean, tmp16_m2, tmp16_weight, roffset == 0
        )
        tmp16_mean = tl.where(rmask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(rmask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(rmask & xmask, tmp16_weight_next, tmp16_weight)
        tl.store(in_out_ptr0 + (r2 + (2560*x3)), tmp14, rmask & xmask)
    tmp16_tmp, tmp17_tmp, tmp18_tmp = triton_helpers.welford(
        tmp16_mean, tmp16_m2, tmp16_weight, 1
    )
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_out_ptr0 + (r2 + (2560*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp19 - tmp16
        tmp21 = 2560.0
        tmp22 = tmp17 / tmp21
        tmp23 = 1e-05
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp26 * tmp27
        tmp30 = tmp28 + tmp29
        tl.store(out_ptr2 + (r2 + (2560*x3)), tmp30, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x2/cx2l5v2odxmc3fzns5d7eywjhhijtme4miqh4ebrtbgtf7n7ebog.py
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
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 10240
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


# kernel path: /tmp/torchinductor_sahanp/kl/ckl3ycmyy4m2beub4etmvpqersc6re7avt6i6vyojjmhidqdgrwq.py
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
triton_red_fused_add_native_layer_norm_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
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
        tmp0 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
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
        tmp9 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 2560.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (2560*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ax/cax4cbrw5iokdytra7h2vcmu637epn5ne26szgu5ovctml6cg4ct.py
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
triton_red_fused_add_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r1 + (2560*x0)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp10
        tmp15 = 2560.0
        tmp16 = tmp11 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r1 + (2560*x0)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yr/cyrbtmxhm5nfoukvjrdcwaytvxt4nhxe67s66u55dgrnlahplnpe.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_24, exp_24, sub_73, sum_25
# Graph fragment:
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_532, [1], True), kwargs = {})
#   %sub_73 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_532, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_73,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8008
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
        tmp0 = tl.load(in_ptr0 + (r1 + (8032*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (8032*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yv/cyvcdbcui2dq72wob2pph2ojcfkqy4ynjfsjmjjerbktmbgzu4wi.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type, div_24, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_2
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_533, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_533, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type), kwargs = {})
triton_per_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp11 = tl.load(in_ptr2 + (r0), None)
    tmp13 = tl.load(in_ptr3 + (r0), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.full([RBLOCK], 8008, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < 8008), "index out of bounds: 0 <= tmp8 < 8008")
    tmp10 = tl.load(in_ptr1 + (tmp8 + (8032*r0)), None, eviction_policy='evict_last')
    tmp12 = tmp10 - tmp11
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp12 - tmp14
    tmp16 = -tmp15
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp2.to(tl.int64)
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp21 / tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 128), (128, 1))
    assert_size_stride(arg1_1, (8008, 2560), (2560, 1))
    assert_size_stride(arg2_1, (128, 2560), (2560, 1))
    assert_size_stride(arg3_1, (2560, ), (1, ))
    assert_size_stride(arg4_1, (2560, ), (1, ))
    assert_size_stride(arg5_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg6_1, (2560, ), (1, ))
    assert_size_stride(arg7_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg8_1, (2560, ), (1, ))
    assert_size_stride(arg9_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg10_1, (2560, ), (1, ))
    assert_size_stride(arg11_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg12_1, (2560, ), (1, ))
    assert_size_stride(arg13_1, (2560, ), (1, ))
    assert_size_stride(arg14_1, (2560, ), (1, ))
    assert_size_stride(arg15_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg16_1, (10240, ), (1, ))
    assert_size_stride(arg17_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg18_1, (2560, ), (1, ))
    assert_size_stride(arg19_1, (2560, ), (1, ))
    assert_size_stride(arg20_1, (2560, ), (1, ))
    assert_size_stride(arg21_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg22_1, (2560, ), (1, ))
    assert_size_stride(arg23_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg24_1, (2560, ), (1, ))
    assert_size_stride(arg25_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg26_1, (2560, ), (1, ))
    assert_size_stride(arg27_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg28_1, (2560, ), (1, ))
    assert_size_stride(arg29_1, (2560, ), (1, ))
    assert_size_stride(arg30_1, (2560, ), (1, ))
    assert_size_stride(arg31_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg32_1, (10240, ), (1, ))
    assert_size_stride(arg33_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg34_1, (2560, ), (1, ))
    assert_size_stride(arg35_1, (2560, ), (1, ))
    assert_size_stride(arg36_1, (2560, ), (1, ))
    assert_size_stride(arg37_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg38_1, (2560, ), (1, ))
    assert_size_stride(arg39_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg40_1, (2560, ), (1, ))
    assert_size_stride(arg41_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg42_1, (2560, ), (1, ))
    assert_size_stride(arg43_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg44_1, (2560, ), (1, ))
    assert_size_stride(arg45_1, (2560, ), (1, ))
    assert_size_stride(arg46_1, (2560, ), (1, ))
    assert_size_stride(arg47_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg48_1, (10240, ), (1, ))
    assert_size_stride(arg49_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg50_1, (2560, ), (1, ))
    assert_size_stride(arg51_1, (2560, ), (1, ))
    assert_size_stride(arg52_1, (2560, ), (1, ))
    assert_size_stride(arg53_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg54_1, (2560, ), (1, ))
    assert_size_stride(arg55_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg56_1, (2560, ), (1, ))
    assert_size_stride(arg57_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg58_1, (2560, ), (1, ))
    assert_size_stride(arg59_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg60_1, (2560, ), (1, ))
    assert_size_stride(arg61_1, (2560, ), (1, ))
    assert_size_stride(arg62_1, (2560, ), (1, ))
    assert_size_stride(arg63_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg64_1, (10240, ), (1, ))
    assert_size_stride(arg65_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg66_1, (2560, ), (1, ))
    assert_size_stride(arg67_1, (2560, ), (1, ))
    assert_size_stride(arg68_1, (2560, ), (1, ))
    assert_size_stride(arg69_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg70_1, (2560, ), (1, ))
    assert_size_stride(arg71_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg72_1, (2560, ), (1, ))
    assert_size_stride(arg73_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg74_1, (2560, ), (1, ))
    assert_size_stride(arg75_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg76_1, (2560, ), (1, ))
    assert_size_stride(arg77_1, (2560, ), (1, ))
    assert_size_stride(arg78_1, (2560, ), (1, ))
    assert_size_stride(arg79_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg80_1, (10240, ), (1, ))
    assert_size_stride(arg81_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg82_1, (2560, ), (1, ))
    assert_size_stride(arg83_1, (2560, ), (1, ))
    assert_size_stride(arg84_1, (2560, ), (1, ))
    assert_size_stride(arg85_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg86_1, (2560, ), (1, ))
    assert_size_stride(arg87_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg88_1, (2560, ), (1, ))
    assert_size_stride(arg89_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg90_1, (2560, ), (1, ))
    assert_size_stride(arg91_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg92_1, (2560, ), (1, ))
    assert_size_stride(arg93_1, (2560, ), (1, ))
    assert_size_stride(arg94_1, (2560, ), (1, ))
    assert_size_stride(arg95_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg96_1, (10240, ), (1, ))
    assert_size_stride(arg97_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg98_1, (2560, ), (1, ))
    assert_size_stride(arg99_1, (2560, ), (1, ))
    assert_size_stride(arg100_1, (2560, ), (1, ))
    assert_size_stride(arg101_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg102_1, (2560, ), (1, ))
    assert_size_stride(arg103_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg104_1, (2560, ), (1, ))
    assert_size_stride(arg105_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg106_1, (2560, ), (1, ))
    assert_size_stride(arg107_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg108_1, (2560, ), (1, ))
    assert_size_stride(arg109_1, (2560, ), (1, ))
    assert_size_stride(arg110_1, (2560, ), (1, ))
    assert_size_stride(arg111_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg112_1, (10240, ), (1, ))
    assert_size_stride(arg113_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg114_1, (2560, ), (1, ))
    assert_size_stride(arg115_1, (2560, ), (1, ))
    assert_size_stride(arg116_1, (2560, ), (1, ))
    assert_size_stride(arg117_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg118_1, (2560, ), (1, ))
    assert_size_stride(arg119_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg120_1, (2560, ), (1, ))
    assert_size_stride(arg121_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg122_1, (2560, ), (1, ))
    assert_size_stride(arg123_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg124_1, (2560, ), (1, ))
    assert_size_stride(arg125_1, (2560, ), (1, ))
    assert_size_stride(arg126_1, (2560, ), (1, ))
    assert_size_stride(arg127_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg128_1, (10240, ), (1, ))
    assert_size_stride(arg129_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg130_1, (2560, ), (1, ))
    assert_size_stride(arg131_1, (2560, ), (1, ))
    assert_size_stride(arg132_1, (2560, ), (1, ))
    assert_size_stride(arg133_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg134_1, (2560, ), (1, ))
    assert_size_stride(arg135_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg136_1, (2560, ), (1, ))
    assert_size_stride(arg137_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg138_1, (2560, ), (1, ))
    assert_size_stride(arg139_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg140_1, (2560, ), (1, ))
    assert_size_stride(arg141_1, (2560, ), (1, ))
    assert_size_stride(arg142_1, (2560, ), (1, ))
    assert_size_stride(arg143_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg144_1, (10240, ), (1, ))
    assert_size_stride(arg145_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg146_1, (2560, ), (1, ))
    assert_size_stride(arg147_1, (2560, ), (1, ))
    assert_size_stride(arg148_1, (2560, ), (1, ))
    assert_size_stride(arg149_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg150_1, (2560, ), (1, ))
    assert_size_stride(arg151_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg152_1, (2560, ), (1, ))
    assert_size_stride(arg153_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg154_1, (2560, ), (1, ))
    assert_size_stride(arg155_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg156_1, (2560, ), (1, ))
    assert_size_stride(arg157_1, (2560, ), (1, ))
    assert_size_stride(arg158_1, (2560, ), (1, ))
    assert_size_stride(arg159_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg160_1, (10240, ), (1, ))
    assert_size_stride(arg161_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg162_1, (2560, ), (1, ))
    assert_size_stride(arg163_1, (2560, ), (1, ))
    assert_size_stride(arg164_1, (2560, ), (1, ))
    assert_size_stride(arg165_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg166_1, (2560, ), (1, ))
    assert_size_stride(arg167_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg168_1, (2560, ), (1, ))
    assert_size_stride(arg169_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg170_1, (2560, ), (1, ))
    assert_size_stride(arg171_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg172_1, (2560, ), (1, ))
    assert_size_stride(arg173_1, (2560, ), (1, ))
    assert_size_stride(arg174_1, (2560, ), (1, ))
    assert_size_stride(arg175_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg176_1, (10240, ), (1, ))
    assert_size_stride(arg177_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg178_1, (2560, ), (1, ))
    assert_size_stride(arg179_1, (2560, ), (1, ))
    assert_size_stride(arg180_1, (2560, ), (1, ))
    assert_size_stride(arg181_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg182_1, (2560, ), (1, ))
    assert_size_stride(arg183_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg184_1, (2560, ), (1, ))
    assert_size_stride(arg185_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg186_1, (2560, ), (1, ))
    assert_size_stride(arg187_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg188_1, (2560, ), (1, ))
    assert_size_stride(arg189_1, (2560, ), (1, ))
    assert_size_stride(arg190_1, (2560, ), (1, ))
    assert_size_stride(arg191_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg192_1, (10240, ), (1, ))
    assert_size_stride(arg193_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg194_1, (2560, ), (1, ))
    assert_size_stride(arg195_1, (2560, ), (1, ))
    assert_size_stride(arg196_1, (2560, ), (1, ))
    assert_size_stride(arg197_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg198_1, (2560, ), (1, ))
    assert_size_stride(arg199_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg200_1, (2560, ), (1, ))
    assert_size_stride(arg201_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg202_1, (2560, ), (1, ))
    assert_size_stride(arg203_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg204_1, (2560, ), (1, ))
    assert_size_stride(arg205_1, (2560, ), (1, ))
    assert_size_stride(arg206_1, (2560, ), (1, ))
    assert_size_stride(arg207_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg208_1, (10240, ), (1, ))
    assert_size_stride(arg209_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg210_1, (2560, ), (1, ))
    assert_size_stride(arg211_1, (2560, ), (1, ))
    assert_size_stride(arg212_1, (2560, ), (1, ))
    assert_size_stride(arg213_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg214_1, (2560, ), (1, ))
    assert_size_stride(arg215_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg216_1, (2560, ), (1, ))
    assert_size_stride(arg217_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg218_1, (2560, ), (1, ))
    assert_size_stride(arg219_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg220_1, (2560, ), (1, ))
    assert_size_stride(arg221_1, (2560, ), (1, ))
    assert_size_stride(arg222_1, (2560, ), (1, ))
    assert_size_stride(arg223_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg224_1, (10240, ), (1, ))
    assert_size_stride(arg225_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg226_1, (2560, ), (1, ))
    assert_size_stride(arg227_1, (2560, ), (1, ))
    assert_size_stride(arg228_1, (2560, ), (1, ))
    assert_size_stride(arg229_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg230_1, (2560, ), (1, ))
    assert_size_stride(arg231_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg232_1, (2560, ), (1, ))
    assert_size_stride(arg233_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg234_1, (2560, ), (1, ))
    assert_size_stride(arg235_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg236_1, (2560, ), (1, ))
    assert_size_stride(arg237_1, (2560, ), (1, ))
    assert_size_stride(arg238_1, (2560, ), (1, ))
    assert_size_stride(arg239_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg240_1, (10240, ), (1, ))
    assert_size_stride(arg241_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg242_1, (2560, ), (1, ))
    assert_size_stride(arg243_1, (2560, ), (1, ))
    assert_size_stride(arg244_1, (2560, ), (1, ))
    assert_size_stride(arg245_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg246_1, (2560, ), (1, ))
    assert_size_stride(arg247_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg248_1, (2560, ), (1, ))
    assert_size_stride(arg249_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg250_1, (2560, ), (1, ))
    assert_size_stride(arg251_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg252_1, (2560, ), (1, ))
    assert_size_stride(arg253_1, (2560, ), (1, ))
    assert_size_stride(arg254_1, (2560, ), (1, ))
    assert_size_stride(arg255_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg256_1, (10240, ), (1, ))
    assert_size_stride(arg257_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg258_1, (2560, ), (1, ))
    assert_size_stride(arg259_1, (2560, ), (1, ))
    assert_size_stride(arg260_1, (2560, ), (1, ))
    assert_size_stride(arg261_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg262_1, (2560, ), (1, ))
    assert_size_stride(arg263_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg264_1, (2560, ), (1, ))
    assert_size_stride(arg265_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg266_1, (2560, ), (1, ))
    assert_size_stride(arg267_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg268_1, (2560, ), (1, ))
    assert_size_stride(arg269_1, (2560, ), (1, ))
    assert_size_stride(arg270_1, (2560, ), (1, ))
    assert_size_stride(arg271_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg272_1, (10240, ), (1, ))
    assert_size_stride(arg273_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg274_1, (2560, ), (1, ))
    assert_size_stride(arg275_1, (2560, ), (1, ))
    assert_size_stride(arg276_1, (2560, ), (1, ))
    assert_size_stride(arg277_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg278_1, (2560, ), (1, ))
    assert_size_stride(arg279_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg280_1, (2560, ), (1, ))
    assert_size_stride(arg281_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg282_1, (2560, ), (1, ))
    assert_size_stride(arg283_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg284_1, (2560, ), (1, ))
    assert_size_stride(arg285_1, (2560, ), (1, ))
    assert_size_stride(arg286_1, (2560, ), (1, ))
    assert_size_stride(arg287_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg288_1, (10240, ), (1, ))
    assert_size_stride(arg289_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg290_1, (2560, ), (1, ))
    assert_size_stride(arg291_1, (2560, ), (1, ))
    assert_size_stride(arg292_1, (2560, ), (1, ))
    assert_size_stride(arg293_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg294_1, (2560, ), (1, ))
    assert_size_stride(arg295_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg296_1, (2560, ), (1, ))
    assert_size_stride(arg297_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg298_1, (2560, ), (1, ))
    assert_size_stride(arg299_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg300_1, (2560, ), (1, ))
    assert_size_stride(arg301_1, (2560, ), (1, ))
    assert_size_stride(arg302_1, (2560, ), (1, ))
    assert_size_stride(arg303_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg304_1, (10240, ), (1, ))
    assert_size_stride(arg305_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg306_1, (2560, ), (1, ))
    assert_size_stride(arg307_1, (2560, ), (1, ))
    assert_size_stride(arg308_1, (2560, ), (1, ))
    assert_size_stride(arg309_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg310_1, (2560, ), (1, ))
    assert_size_stride(arg311_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg312_1, (2560, ), (1, ))
    assert_size_stride(arg313_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg314_1, (2560, ), (1, ))
    assert_size_stride(arg315_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg316_1, (2560, ), (1, ))
    assert_size_stride(arg317_1, (2560, ), (1, ))
    assert_size_stride(arg318_1, (2560, ), (1, ))
    assert_size_stride(arg319_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg320_1, (10240, ), (1, ))
    assert_size_stride(arg321_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg322_1, (2560, ), (1, ))
    assert_size_stride(arg323_1, (2560, ), (1, ))
    assert_size_stride(arg324_1, (2560, ), (1, ))
    assert_size_stride(arg325_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg326_1, (2560, ), (1, ))
    assert_size_stride(arg327_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg328_1, (2560, ), (1, ))
    assert_size_stride(arg329_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg330_1, (2560, ), (1, ))
    assert_size_stride(arg331_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg332_1, (2560, ), (1, ))
    assert_size_stride(arg333_1, (2560, ), (1, ))
    assert_size_stride(arg334_1, (2560, ), (1, ))
    assert_size_stride(arg335_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg336_1, (10240, ), (1, ))
    assert_size_stride(arg337_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg338_1, (2560, ), (1, ))
    assert_size_stride(arg339_1, (2560, ), (1, ))
    assert_size_stride(arg340_1, (2560, ), (1, ))
    assert_size_stride(arg341_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg342_1, (2560, ), (1, ))
    assert_size_stride(arg343_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg344_1, (2560, ), (1, ))
    assert_size_stride(arg345_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg346_1, (2560, ), (1, ))
    assert_size_stride(arg347_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg348_1, (2560, ), (1, ))
    assert_size_stride(arg349_1, (2560, ), (1, ))
    assert_size_stride(arg350_1, (2560, ), (1, ))
    assert_size_stride(arg351_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg352_1, (10240, ), (1, ))
    assert_size_stride(arg353_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg354_1, (2560, ), (1, ))
    assert_size_stride(arg355_1, (2560, ), (1, ))
    assert_size_stride(arg356_1, (2560, ), (1, ))
    assert_size_stride(arg357_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg358_1, (2560, ), (1, ))
    assert_size_stride(arg359_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg360_1, (2560, ), (1, ))
    assert_size_stride(arg361_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg362_1, (2560, ), (1, ))
    assert_size_stride(arg363_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg364_1, (2560, ), (1, ))
    assert_size_stride(arg365_1, (2560, ), (1, ))
    assert_size_stride(arg366_1, (2560, ), (1, ))
    assert_size_stride(arg367_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg368_1, (10240, ), (1, ))
    assert_size_stride(arg369_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg370_1, (2560, ), (1, ))
    assert_size_stride(arg371_1, (2560, ), (1, ))
    assert_size_stride(arg372_1, (2560, ), (1, ))
    assert_size_stride(arg373_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg374_1, (2560, ), (1, ))
    assert_size_stride(arg375_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg376_1, (2560, ), (1, ))
    assert_size_stride(arg377_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg378_1, (2560, ), (1, ))
    assert_size_stride(arg379_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg380_1, (2560, ), (1, ))
    assert_size_stride(arg381_1, (2560, ), (1, ))
    assert_size_stride(arg382_1, (2560, ), (1, ))
    assert_size_stride(arg383_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg384_1, (10240, ), (1, ))
    assert_size_stride(arg385_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg386_1, (2560, ), (1, ))
    assert_size_stride(arg387_1, (2560, ), (1, ))
    assert_size_stride(arg388_1, (2560, ), (1, ))
    assert_size_stride(arg389_1, (4, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((4, 128, 2560), (327680, 2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf3, 512, 2560, grid=grid(512), stream=stream0)
        del arg3_1
        del arg4_1
        buf4 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg5_1, (2560, 2560), (1, 2560), 0), out=buf4)
        del arg5_1
        buf5 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg7_1, (2560, 2560), (1, 2560), 0), out=buf5)
        del arg7_1
        buf6 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf5, arg8_1, buf6, 1310720, grid=grid(1310720), stream=stream0)
        del arg8_1
        buf7 = reinterpret_tensor(buf5, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf4, arg6_1, buf7, 1310720, grid=grid(1310720), stream=stream0)
        del arg6_1
        buf8 = empty_strided_cuda((128, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf6, (128, 80, 128), (10240, 1, 80), 0), out=buf8)
        buf13 = empty_strided_cuda((128, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf8, buf13, 16384, 128, grid=grid(16384), stream=stream0)
        buf11 = reinterpret_tensor(buf7, (512, 2560), (2560, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg9_1, (2560, 2560), (1, 2560), 0), out=buf11)
        del arg9_1
        buf12 = reinterpret_tensor(buf3, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf11, arg10_1, buf12, 1310720, grid=grid(1310720), stream=stream0)
        del arg10_1
        buf14 = reinterpret_tensor(buf11, (128, 128, 80), (10240, 80, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_3, attn_output], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (128, 128, 80), (10240, 80, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf4, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf14, buf15, 1310720, grid=grid(1310720), stream=stream0)
        buf16 = reinterpret_tensor(buf14, (512, 2560), (2560, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg11_1, (2560, 2560), (1, 2560), 0), out=buf16)
        del arg11_1
        buf17 = reinterpret_tensor(buf16, (4, 128, 2560), (327680, 2560, 1), 0); del buf16  # reuse
        buf21 = reinterpret_tensor(buf15, (4, 128, 2560), (327680, 2560, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, positions, positions_1, hidden_states, hidden_states_4, hidden_states_5], Original ATen: [aten.embedding, aten.mul, aten.arange, aten.add, aten.native_layer_norm]
        triton_red_fused_add_arange_embedding_mul_native_layer_norm_5.run(buf17, arg0_1, arg1_1, arg2_1, arg12_1, arg13_1, arg14_1, buf21, 512, 2560, grid=grid(512), stream=stream0)
        del arg0_1
        del arg12_1
        del arg13_1
        del arg14_1
        del arg2_1
        buf22 = empty_strided_cuda((512, 10240), (10240, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg15_1, (2560, 10240), (1, 2560), 0), out=buf22)
        del arg15_1
        buf23 = reinterpret_tensor(buf22, (4, 128, 10240), (1310720, 10240, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf23, arg16_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg16_1
        buf24 = reinterpret_tensor(buf21, (512, 2560), (2560, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg17_1, (10240, 2560), (1, 10240), 0), out=buf24)
        del arg17_1
        buf28 = empty_strided_cuda((4, 128, 2560), (327680, 2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf17, buf24, arg18_1, arg19_1, arg20_1, buf28, 512, 2560, grid=grid(512), stream=stream0)
        del arg19_1
        del arg20_1
        buf29 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg21_1, (2560, 2560), (1, 2560), 0), out=buf29)
        del arg21_1
        buf30 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg23_1, (2560, 2560), (1, 2560), 0), out=buf30)
        del arg23_1
        buf31 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf30, arg24_1, buf31, 1310720, grid=grid(1310720), stream=stream0)
        del arg24_1
        buf32 = reinterpret_tensor(buf30, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf29, arg22_1, buf32, 1310720, grid=grid(1310720), stream=stream0)
        del arg22_1
        buf33 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf31, (128, 80, 128), (10240, 1, 80), 0), out=buf33)
        buf38 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf33, buf38, 16384, 128, grid=grid(16384), stream=stream0)
        buf36 = reinterpret_tensor(buf32, (512, 2560), (2560, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf28, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg25_1, (2560, 2560), (1, 2560), 0), out=buf36)
        del arg25_1
        buf37 = reinterpret_tensor(buf28, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf36, arg26_1, buf37, 1310720, grid=grid(1310720), stream=stream0)
        del arg26_1
        buf39 = reinterpret_tensor(buf36, (128, 128, 80), (10240, 80, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7, attn_output_5], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf38, reinterpret_tensor(buf37, (128, 128, 80), (10240, 80, 1), 0), out=buf39)
        buf40 = reinterpret_tensor(buf29, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf39, buf40, 1310720, grid=grid(1310720), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (512, 2560), (2560, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg27_1, (2560, 2560), (1, 2560), 0), out=buf41)
        del arg27_1
        buf42 = reinterpret_tensor(buf41, (4, 128, 2560), (327680, 2560, 1), 0); del buf41  # reuse
        buf46 = reinterpret_tensor(buf40, (4, 128, 2560), (327680, 2560, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_13, hidden_states_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf42, buf17, buf24, arg18_1, arg28_1, arg29_1, arg30_1, buf46, 512, 2560, grid=grid(512), stream=stream0)
        del arg18_1
        del arg28_1
        del arg29_1
        del arg30_1
        buf47 = reinterpret_tensor(buf23, (512, 10240), (10240, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg31_1, (2560, 10240), (1, 2560), 0), out=buf47)
        del arg31_1
        buf48 = reinterpret_tensor(buf47, (4, 128, 10240), (1310720, 10240, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf48, arg32_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg32_1
        buf49 = reinterpret_tensor(buf46, (512, 2560), (2560, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg33_1, (10240, 2560), (1, 10240), 0), out=buf49)
        del arg33_1
        buf53 = reinterpret_tensor(buf24, (4, 128, 2560), (327680, 2560, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf42, buf49, arg34_1, arg35_1, arg36_1, buf53, 512, 2560, grid=grid(512), stream=stream0)
        del arg35_1
        del arg36_1
        buf54 = reinterpret_tensor(buf17, (512, 2560), (2560, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg37_1, (2560, 2560), (1, 2560), 0), out=buf54)
        del arg37_1
        buf55 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg39_1, (2560, 2560), (1, 2560), 0), out=buf55)
        del arg39_1
        buf56 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf55, arg40_1, buf56, 1310720, grid=grid(1310720), stream=stream0)
        del arg40_1
        buf57 = reinterpret_tensor(buf55, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf54, arg38_1, buf57, 1310720, grid=grid(1310720), stream=stream0)
        del arg38_1
        buf58 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf56, (128, 80, 128), (10240, 1, 80), 0), out=buf58)
        buf63 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf58, buf63, 16384, 128, grid=grid(16384), stream=stream0)
        buf61 = reinterpret_tensor(buf57, (512, 2560), (2560, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf53, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg41_1, (2560, 2560), (1, 2560), 0), out=buf61)
        del arg41_1
        buf62 = reinterpret_tensor(buf53, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf61, arg42_1, buf62, 1310720, grid=grid(1310720), stream=stream0)
        del arg42_1
        buf64 = reinterpret_tensor(buf61, (128, 128, 80), (10240, 80, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11, attn_output_10], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf63, reinterpret_tensor(buf62, (128, 128, 80), (10240, 80, 1), 0), out=buf64)
        buf65 = reinterpret_tensor(buf54, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf64, buf65, 1310720, grid=grid(1310720), stream=stream0)
        buf66 = reinterpret_tensor(buf64, (512, 2560), (2560, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg43_1, (2560, 2560), (1, 2560), 0), out=buf66)
        del arg43_1
        buf67 = reinterpret_tensor(buf66, (4, 128, 2560), (327680, 2560, 1), 0); del buf66  # reuse
        buf71 = reinterpret_tensor(buf65, (4, 128, 2560), (327680, 2560, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_22, hidden_states_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf67, buf42, buf49, arg34_1, arg44_1, arg45_1, arg46_1, buf71, 512, 2560, grid=grid(512), stream=stream0)
        del arg34_1
        del arg44_1
        del arg45_1
        del arg46_1
        buf72 = reinterpret_tensor(buf48, (512, 10240), (10240, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg47_1, (2560, 10240), (1, 2560), 0), out=buf72)
        del arg47_1
        buf73 = reinterpret_tensor(buf72, (4, 128, 10240), (1310720, 10240, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_24], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf73, arg48_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg48_1
        buf74 = reinterpret_tensor(buf71, (512, 2560), (2560, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg49_1, (10240, 2560), (1, 10240), 0), out=buf74)
        del arg49_1
        buf78 = reinterpret_tensor(buf49, (4, 128, 2560), (327680, 2560, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf67, buf74, arg50_1, arg51_1, arg52_1, buf78, 512, 2560, grid=grid(512), stream=stream0)
        del arg51_1
        del arg52_1
        buf79 = reinterpret_tensor(buf42, (512, 2560), (2560, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg53_1, (2560, 2560), (1, 2560), 0), out=buf79)
        del arg53_1
        buf80 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg55_1, (2560, 2560), (1, 2560), 0), out=buf80)
        del arg55_1
        buf81 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf80, arg56_1, buf81, 1310720, grid=grid(1310720), stream=stream0)
        del arg56_1
        buf82 = reinterpret_tensor(buf80, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf79, arg54_1, buf82, 1310720, grid=grid(1310720), stream=stream0)
        del arg54_1
        buf83 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf82, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf81, (128, 80, 128), (10240, 1, 80), 0), out=buf83)
        buf88 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf83, buf88, 16384, 128, grid=grid(16384), stream=stream0)
        buf86 = reinterpret_tensor(buf82, (512, 2560), (2560, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg57_1, (2560, 2560), (1, 2560), 0), out=buf86)
        del arg57_1
        buf87 = reinterpret_tensor(buf78, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf86, arg58_1, buf87, 1310720, grid=grid(1310720), stream=stream0)
        del arg58_1
        buf89 = reinterpret_tensor(buf86, (128, 128, 80), (10240, 80, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15, attn_output_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf88, reinterpret_tensor(buf87, (128, 128, 80), (10240, 80, 1), 0), out=buf89)
        buf90 = reinterpret_tensor(buf79, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf90, 1310720, grid=grid(1310720), stream=stream0)
        buf91 = reinterpret_tensor(buf89, (512, 2560), (2560, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg59_1, (2560, 2560), (1, 2560), 0), out=buf91)
        del arg59_1
        buf92 = reinterpret_tensor(buf91, (4, 128, 2560), (327680, 2560, 1), 0); del buf91  # reuse
        buf96 = reinterpret_tensor(buf90, (4, 128, 2560), (327680, 2560, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_31, hidden_states_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf92, buf67, buf74, arg50_1, arg60_1, arg61_1, arg62_1, buf96, 512, 2560, grid=grid(512), stream=stream0)
        del arg50_1
        del arg60_1
        del arg61_1
        del arg62_1
        buf97 = reinterpret_tensor(buf73, (512, 10240), (10240, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg63_1, (2560, 10240), (1, 2560), 0), out=buf97)
        del arg63_1
        buf98 = reinterpret_tensor(buf97, (4, 128, 10240), (1310720, 10240, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf98, arg64_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg64_1
        buf99 = reinterpret_tensor(buf96, (512, 2560), (2560, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg65_1, (10240, 2560), (1, 10240), 0), out=buf99)
        del arg65_1
        buf103 = reinterpret_tensor(buf74, (4, 128, 2560), (327680, 2560, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf92, buf99, arg66_1, arg67_1, arg68_1, buf103, 512, 2560, grid=grid(512), stream=stream0)
        del arg67_1
        del arg68_1
        buf104 = reinterpret_tensor(buf67, (512, 2560), (2560, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg69_1, (2560, 2560), (1, 2560), 0), out=buf104)
        del arg69_1
        buf105 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg71_1, (2560, 2560), (1, 2560), 0), out=buf105)
        del arg71_1
        buf106 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf105, arg72_1, buf106, 1310720, grid=grid(1310720), stream=stream0)
        del arg72_1
        buf107 = reinterpret_tensor(buf105, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf104, arg70_1, buf107, 1310720, grid=grid(1310720), stream=stream0)
        del arg70_1
        buf108 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf106, (128, 80, 128), (10240, 1, 80), 0), out=buf108)
        buf113 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf108, buf113, 16384, 128, grid=grid(16384), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (512, 2560), (2560, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg73_1, (2560, 2560), (1, 2560), 0), out=buf111)
        del arg73_1
        buf112 = reinterpret_tensor(buf103, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf111, arg74_1, buf112, 1310720, grid=grid(1310720), stream=stream0)
        del arg74_1
        buf114 = reinterpret_tensor(buf111, (128, 128, 80), (10240, 80, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19, attn_output_20], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf113, reinterpret_tensor(buf112, (128, 128, 80), (10240, 80, 1), 0), out=buf114)
        buf115 = reinterpret_tensor(buf104, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf114, buf115, 1310720, grid=grid(1310720), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (512, 2560), (2560, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf115, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg75_1, (2560, 2560), (1, 2560), 0), out=buf116)
        del arg75_1
        buf117 = reinterpret_tensor(buf116, (4, 128, 2560), (327680, 2560, 1), 0); del buf116  # reuse
        buf121 = reinterpret_tensor(buf115, (4, 128, 2560), (327680, 2560, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37, hidden_states_40, hidden_states_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf117, buf92, buf99, arg66_1, arg76_1, arg77_1, arg78_1, buf121, 512, 2560, grid=grid(512), stream=stream0)
        del arg66_1
        del arg76_1
        del arg77_1
        del arg78_1
        buf122 = reinterpret_tensor(buf98, (512, 10240), (10240, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg79_1, (2560, 10240), (1, 2560), 0), out=buf122)
        del arg79_1
        buf123 = reinterpret_tensor(buf122, (4, 128, 10240), (1310720, 10240, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf123, arg80_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg80_1
        buf124 = reinterpret_tensor(buf121, (512, 2560), (2560, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg81_1, (10240, 2560), (1, 10240), 0), out=buf124)
        del arg81_1
        buf128 = reinterpret_tensor(buf99, (4, 128, 2560), (327680, 2560, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf117, buf124, arg82_1, arg83_1, arg84_1, buf128, 512, 2560, grid=grid(512), stream=stream0)
        del arg83_1
        del arg84_1
        buf129 = reinterpret_tensor(buf92, (512, 2560), (2560, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg85_1, (2560, 2560), (1, 2560), 0), out=buf129)
        del arg85_1
        buf130 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg87_1, (2560, 2560), (1, 2560), 0), out=buf130)
        del arg87_1
        buf131 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf130, arg88_1, buf131, 1310720, grid=grid(1310720), stream=stream0)
        del arg88_1
        buf132 = reinterpret_tensor(buf130, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, arg86_1, buf132, 1310720, grid=grid(1310720), stream=stream0)
        del arg86_1
        buf133 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf131, (128, 80, 128), (10240, 1, 80), 0), out=buf133)
        buf138 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf133, buf138, 16384, 128, grid=grid(16384), stream=stream0)
        buf136 = reinterpret_tensor(buf132, (512, 2560), (2560, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg89_1, (2560, 2560), (1, 2560), 0), out=buf136)
        del arg89_1
        buf137 = reinterpret_tensor(buf128, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf136, arg90_1, buf137, 1310720, grid=grid(1310720), stream=stream0)
        del arg90_1
        buf139 = reinterpret_tensor(buf136, (128, 128, 80), (10240, 80, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23, attn_output_25], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf138, reinterpret_tensor(buf137, (128, 128, 80), (10240, 80, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf129, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf139, buf140, 1310720, grid=grid(1310720), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (512, 2560), (2560, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg91_1, (2560, 2560), (1, 2560), 0), out=buf141)
        del arg91_1
        buf142 = reinterpret_tensor(buf141, (4, 128, 2560), (327680, 2560, 1), 0); del buf141  # reuse
        buf146 = reinterpret_tensor(buf140, (4, 128, 2560), (327680, 2560, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_49, hidden_states_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf142, buf117, buf124, arg82_1, arg92_1, arg93_1, arg94_1, buf146, 512, 2560, grid=grid(512), stream=stream0)
        del arg82_1
        del arg92_1
        del arg93_1
        del arg94_1
        buf147 = reinterpret_tensor(buf123, (512, 10240), (10240, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg95_1, (2560, 10240), (1, 2560), 0), out=buf147)
        del arg95_1
        buf148 = reinterpret_tensor(buf147, (4, 128, 10240), (1310720, 10240, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf148, arg96_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg96_1
        buf149 = reinterpret_tensor(buf146, (512, 2560), (2560, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg97_1, (10240, 2560), (1, 10240), 0), out=buf149)
        del arg97_1
        buf153 = reinterpret_tensor(buf124, (4, 128, 2560), (327680, 2560, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf142, buf149, arg98_1, arg99_1, arg100_1, buf153, 512, 2560, grid=grid(512), stream=stream0)
        del arg100_1
        del arg99_1
        buf154 = reinterpret_tensor(buf117, (512, 2560), (2560, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg101_1, (2560, 2560), (1, 2560), 0), out=buf154)
        del arg101_1
        buf155 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg103_1, (2560, 2560), (1, 2560), 0), out=buf155)
        del arg103_1
        buf156 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf155, arg104_1, buf156, 1310720, grid=grid(1310720), stream=stream0)
        del arg104_1
        buf157 = reinterpret_tensor(buf155, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [contiguous_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf154, arg102_1, buf157, 1310720, grid=grid(1310720), stream=stream0)
        del arg102_1
        buf158 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf156, (128, 80, 128), (10240, 1, 80), 0), out=buf158)
        buf163 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf158, buf163, 16384, 128, grid=grid(16384), stream=stream0)
        buf161 = reinterpret_tensor(buf157, (512, 2560), (2560, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg105_1, (2560, 2560), (1, 2560), 0), out=buf161)
        del arg105_1
        buf162 = reinterpret_tensor(buf153, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [value_states_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf161, arg106_1, buf162, 1310720, grid=grid(1310720), stream=stream0)
        del arg106_1
        buf164 = reinterpret_tensor(buf161, (128, 128, 80), (10240, 80, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_27, attn_output_30], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf163, reinterpret_tensor(buf162, (128, 128, 80), (10240, 80, 1), 0), out=buf164)
        buf165 = reinterpret_tensor(buf154, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf164, buf165, 1310720, grid=grid(1310720), stream=stream0)
        buf166 = reinterpret_tensor(buf164, (512, 2560), (2560, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg107_1, (2560, 2560), (1, 2560), 0), out=buf166)
        del arg107_1
        buf167 = reinterpret_tensor(buf166, (4, 128, 2560), (327680, 2560, 1), 0); del buf166  # reuse
        buf171 = reinterpret_tensor(buf165, (4, 128, 2560), (327680, 2560, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_58, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf167, buf142, buf149, arg98_1, arg108_1, arg109_1, arg110_1, buf171, 512, 2560, grid=grid(512), stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del arg98_1
        buf172 = reinterpret_tensor(buf148, (512, 10240), (10240, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf171, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg111_1, (2560, 10240), (1, 2560), 0), out=buf172)
        del arg111_1
        buf173 = reinterpret_tensor(buf172, (4, 128, 10240), (1310720, 10240, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_60], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf173, arg112_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg112_1
        buf174 = reinterpret_tensor(buf171, (512, 2560), (2560, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf173, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg113_1, (10240, 2560), (1, 10240), 0), out=buf174)
        del arg113_1
        buf178 = reinterpret_tensor(buf149, (4, 128, 2560), (327680, 2560, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf167, buf174, arg114_1, arg115_1, arg116_1, buf178, 512, 2560, grid=grid(512), stream=stream0)
        del arg115_1
        del arg116_1
        buf179 = reinterpret_tensor(buf142, (512, 2560), (2560, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg117_1, (2560, 2560), (1, 2560), 0), out=buf179)
        del arg117_1
        buf180 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg119_1, (2560, 2560), (1, 2560), 0), out=buf180)
        del arg119_1
        buf181 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf180, arg120_1, buf181, 1310720, grid=grid(1310720), stream=stream0)
        del arg120_1
        buf182 = reinterpret_tensor(buf180, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [contiguous_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf179, arg118_1, buf182, 1310720, grid=grid(1310720), stream=stream0)
        del arg118_1
        buf183 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf181, (128, 80, 128), (10240, 1, 80), 0), out=buf183)
        buf188 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf183, buf188, 16384, 128, grid=grid(16384), stream=stream0)
        buf186 = reinterpret_tensor(buf182, (512, 2560), (2560, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg121_1, (2560, 2560), (1, 2560), 0), out=buf186)
        del arg121_1
        buf187 = reinterpret_tensor(buf178, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [value_states_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf186, arg122_1, buf187, 1310720, grid=grid(1310720), stream=stream0)
        del arg122_1
        buf189 = reinterpret_tensor(buf186, (128, 128, 80), (10240, 80, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_31, attn_output_35], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf188, reinterpret_tensor(buf187, (128, 128, 80), (10240, 80, 1), 0), out=buf189)
        buf190 = reinterpret_tensor(buf179, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [attn_output_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf189, buf190, 1310720, grid=grid(1310720), stream=stream0)
        buf191 = reinterpret_tensor(buf189, (512, 2560), (2560, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf190, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg123_1, (2560, 2560), (1, 2560), 0), out=buf191)
        del arg123_1
        buf192 = reinterpret_tensor(buf191, (4, 128, 2560), (327680, 2560, 1), 0); del buf191  # reuse
        buf196 = reinterpret_tensor(buf190, (4, 128, 2560), (327680, 2560, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_67, hidden_states_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf192, buf167, buf174, arg114_1, arg124_1, arg125_1, arg126_1, buf196, 512, 2560, grid=grid(512), stream=stream0)
        del arg114_1
        del arg124_1
        del arg125_1
        del arg126_1
        buf197 = reinterpret_tensor(buf173, (512, 10240), (10240, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg127_1, (2560, 10240), (1, 2560), 0), out=buf197)
        del arg127_1
        buf198 = reinterpret_tensor(buf197, (4, 128, 10240), (1310720, 10240, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf198, arg128_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg128_1
        buf199 = reinterpret_tensor(buf196, (512, 2560), (2560, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf198, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg129_1, (10240, 2560), (1, 10240), 0), out=buf199)
        del arg129_1
        buf203 = reinterpret_tensor(buf174, (4, 128, 2560), (327680, 2560, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf192, buf199, arg130_1, arg131_1, arg132_1, buf203, 512, 2560, grid=grid(512), stream=stream0)
        del arg131_1
        del arg132_1
        buf204 = reinterpret_tensor(buf167, (512, 2560), (2560, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg133_1, (2560, 2560), (1, 2560), 0), out=buf204)
        del arg133_1
        buf205 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg135_1, (2560, 2560), (1, 2560), 0), out=buf205)
        del arg135_1
        buf206 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf205, arg136_1, buf206, 1310720, grid=grid(1310720), stream=stream0)
        del arg136_1
        buf207 = reinterpret_tensor(buf205, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [contiguous_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf204, arg134_1, buf207, 1310720, grid=grid(1310720), stream=stream0)
        del arg134_1
        buf208 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf206, (128, 80, 128), (10240, 1, 80), 0), out=buf208)
        buf213 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_35], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf208, buf213, 16384, 128, grid=grid(16384), stream=stream0)
        buf211 = reinterpret_tensor(buf207, (512, 2560), (2560, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg137_1, (2560, 2560), (1, 2560), 0), out=buf211)
        del arg137_1
        buf212 = reinterpret_tensor(buf203, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [value_states_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf211, arg138_1, buf212, 1310720, grid=grid(1310720), stream=stream0)
        del arg138_1
        buf214 = reinterpret_tensor(buf211, (128, 128, 80), (10240, 80, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_35, attn_output_40], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf213, reinterpret_tensor(buf212, (128, 128, 80), (10240, 80, 1), 0), out=buf214)
        buf215 = reinterpret_tensor(buf204, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [attn_output_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf214, buf215, 1310720, grid=grid(1310720), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (512, 2560), (2560, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg139_1, (2560, 2560), (1, 2560), 0), out=buf216)
        del arg139_1
        buf217 = reinterpret_tensor(buf216, (4, 128, 2560), (327680, 2560, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf215, (4, 128, 2560), (327680, 2560, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73, hidden_states_76, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf217, buf192, buf199, arg130_1, arg140_1, arg141_1, arg142_1, buf221, 512, 2560, grid=grid(512), stream=stream0)
        del arg130_1
        del arg140_1
        del arg141_1
        del arg142_1
        buf222 = reinterpret_tensor(buf198, (512, 10240), (10240, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg143_1, (2560, 10240), (1, 2560), 0), out=buf222)
        del arg143_1
        buf223 = reinterpret_tensor(buf222, (4, 128, 10240), (1310720, 10240, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf223, arg144_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg144_1
        buf224 = reinterpret_tensor(buf221, (512, 2560), (2560, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg145_1, (10240, 2560), (1, 10240), 0), out=buf224)
        del arg145_1
        buf228 = reinterpret_tensor(buf199, (4, 128, 2560), (327680, 2560, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf217, buf224, arg146_1, arg147_1, arg148_1, buf228, 512, 2560, grid=grid(512), stream=stream0)
        del arg147_1
        del arg148_1
        buf229 = reinterpret_tensor(buf192, (512, 2560), (2560, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg149_1, (2560, 2560), (1, 2560), 0), out=buf229)
        del arg149_1
        buf230 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg151_1, (2560, 2560), (1, 2560), 0), out=buf230)
        del arg151_1
        buf231 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf230, arg152_1, buf231, 1310720, grid=grid(1310720), stream=stream0)
        del arg152_1
        buf232 = reinterpret_tensor(buf230, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf229, arg150_1, buf232, 1310720, grid=grid(1310720), stream=stream0)
        del arg150_1
        buf233 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf231, (128, 80, 128), (10240, 1, 80), 0), out=buf233)
        buf238 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf233, buf238, 16384, 128, grid=grid(16384), stream=stream0)
        buf236 = reinterpret_tensor(buf232, (512, 2560), (2560, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg153_1, (2560, 2560), (1, 2560), 0), out=buf236)
        del arg153_1
        buf237 = reinterpret_tensor(buf228, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [value_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf236, arg154_1, buf237, 1310720, grid=grid(1310720), stream=stream0)
        del arg154_1
        buf239 = reinterpret_tensor(buf236, (128, 128, 80), (10240, 80, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_39, attn_output_45], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf238, reinterpret_tensor(buf237, (128, 128, 80), (10240, 80, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf229, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [attn_output_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf239, buf240, 1310720, grid=grid(1310720), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (512, 2560), (2560, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg155_1, (2560, 2560), (1, 2560), 0), out=buf241)
        del arg155_1
        buf242 = reinterpret_tensor(buf241, (4, 128, 2560), (327680, 2560, 1), 0); del buf241  # reuse
        buf246 = reinterpret_tensor(buf240, (4, 128, 2560), (327680, 2560, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_85, hidden_states_86], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf242, buf217, buf224, arg146_1, arg156_1, arg157_1, arg158_1, buf246, 512, 2560, grid=grid(512), stream=stream0)
        del arg146_1
        del arg156_1
        del arg157_1
        del arg158_1
        buf247 = reinterpret_tensor(buf223, (512, 10240), (10240, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg159_1, (2560, 10240), (1, 2560), 0), out=buf247)
        del arg159_1
        buf248 = reinterpret_tensor(buf247, (4, 128, 10240), (1310720, 10240, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf248, arg160_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg160_1
        buf249 = reinterpret_tensor(buf246, (512, 2560), (2560, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg161_1, (10240, 2560), (1, 10240), 0), out=buf249)
        del arg161_1
        buf253 = reinterpret_tensor(buf224, (4, 128, 2560), (327680, 2560, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf242, buf249, arg162_1, arg163_1, arg164_1, buf253, 512, 2560, grid=grid(512), stream=stream0)
        del arg163_1
        del arg164_1
        buf254 = reinterpret_tensor(buf217, (512, 2560), (2560, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg165_1, (2560, 2560), (1, 2560), 0), out=buf254)
        del arg165_1
        buf255 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg167_1, (2560, 2560), (1, 2560), 0), out=buf255)
        del arg167_1
        buf256 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf255, arg168_1, buf256, 1310720, grid=grid(1310720), stream=stream0)
        del arg168_1
        buf257 = reinterpret_tensor(buf255, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [contiguous_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf254, arg166_1, buf257, 1310720, grid=grid(1310720), stream=stream0)
        del arg166_1
        buf258 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf256, (128, 80, 128), (10240, 1, 80), 0), out=buf258)
        buf263 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf258, buf263, 16384, 128, grid=grid(16384), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (512, 2560), (2560, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf253, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg169_1, (2560, 2560), (1, 2560), 0), out=buf261)
        del arg169_1
        buf262 = reinterpret_tensor(buf253, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [value_states_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf261, arg170_1, buf262, 1310720, grid=grid(1310720), stream=stream0)
        del arg170_1
        buf264 = reinterpret_tensor(buf261, (128, 128, 80), (10240, 80, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_43, attn_output_50], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf263, reinterpret_tensor(buf262, (128, 128, 80), (10240, 80, 1), 0), out=buf264)
        buf265 = reinterpret_tensor(buf254, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf264, buf265, 1310720, grid=grid(1310720), stream=stream0)
        buf266 = reinterpret_tensor(buf264, (512, 2560), (2560, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg171_1, (2560, 2560), (1, 2560), 0), out=buf266)
        del arg171_1
        buf267 = reinterpret_tensor(buf266, (4, 128, 2560), (327680, 2560, 1), 0); del buf266  # reuse
        buf271 = reinterpret_tensor(buf265, (4, 128, 2560), (327680, 2560, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_94, hidden_states_95], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf267, buf242, buf249, arg162_1, arg172_1, arg173_1, arg174_1, buf271, 512, 2560, grid=grid(512), stream=stream0)
        del arg162_1
        del arg172_1
        del arg173_1
        del arg174_1
        buf272 = reinterpret_tensor(buf248, (512, 10240), (10240, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg175_1, (2560, 10240), (1, 2560), 0), out=buf272)
        del arg175_1
        buf273 = reinterpret_tensor(buf272, (4, 128, 10240), (1310720, 10240, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf273, arg176_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg176_1
        buf274 = reinterpret_tensor(buf271, (512, 2560), (2560, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg177_1, (10240, 2560), (1, 10240), 0), out=buf274)
        del arg177_1
        buf278 = reinterpret_tensor(buf249, (4, 128, 2560), (327680, 2560, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf267, buf274, arg178_1, arg179_1, arg180_1, buf278, 512, 2560, grid=grid(512), stream=stream0)
        del arg179_1
        del arg180_1
        buf279 = reinterpret_tensor(buf242, (512, 2560), (2560, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg181_1, (2560, 2560), (1, 2560), 0), out=buf279)
        del arg181_1
        buf280 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg183_1, (2560, 2560), (1, 2560), 0), out=buf280)
        del arg183_1
        buf281 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf280, arg184_1, buf281, 1310720, grid=grid(1310720), stream=stream0)
        del arg184_1
        buf282 = reinterpret_tensor(buf280, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [contiguous_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf279, arg182_1, buf282, 1310720, grid=grid(1310720), stream=stream0)
        del arg182_1
        buf283 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf282, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf281, (128, 80, 128), (10240, 1, 80), 0), out=buf283)
        buf288 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf283, buf288, 16384, 128, grid=grid(16384), stream=stream0)
        buf286 = reinterpret_tensor(buf282, (512, 2560), (2560, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg185_1, (2560, 2560), (1, 2560), 0), out=buf286)
        del arg185_1
        buf287 = reinterpret_tensor(buf278, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [value_states_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf286, arg186_1, buf287, 1310720, grid=grid(1310720), stream=stream0)
        del arg186_1
        buf289 = reinterpret_tensor(buf286, (128, 128, 80), (10240, 80, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_47, attn_output_55], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf288, reinterpret_tensor(buf287, (128, 128, 80), (10240, 80, 1), 0), out=buf289)
        buf290 = reinterpret_tensor(buf279, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [attn_output_58], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf289, buf290, 1310720, grid=grid(1310720), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (512, 2560), (2560, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf290, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg187_1, (2560, 2560), (1, 2560), 0), out=buf291)
        del arg187_1
        buf292 = reinterpret_tensor(buf291, (4, 128, 2560), (327680, 2560, 1), 0); del buf291  # reuse
        buf296 = reinterpret_tensor(buf290, (4, 128, 2560), (327680, 2560, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_103, hidden_states_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf292, buf267, buf274, arg178_1, arg188_1, arg189_1, arg190_1, buf296, 512, 2560, grid=grid(512), stream=stream0)
        del arg178_1
        del arg188_1
        del arg189_1
        del arg190_1
        buf297 = reinterpret_tensor(buf273, (512, 10240), (10240, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg191_1, (2560, 10240), (1, 2560), 0), out=buf297)
        del arg191_1
        buf298 = reinterpret_tensor(buf297, (4, 128, 10240), (1310720, 10240, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf298, arg192_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg192_1
        buf299 = reinterpret_tensor(buf296, (512, 2560), (2560, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg193_1, (10240, 2560), (1, 10240), 0), out=buf299)
        del arg193_1
        buf303 = reinterpret_tensor(buf274, (4, 128, 2560), (327680, 2560, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_110], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf292, buf299, arg194_1, arg195_1, arg196_1, buf303, 512, 2560, grid=grid(512), stream=stream0)
        del arg195_1
        del arg196_1
        buf304 = reinterpret_tensor(buf267, (512, 2560), (2560, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg197_1, (2560, 2560), (1, 2560), 0), out=buf304)
        del arg197_1
        buf305 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg199_1, (2560, 2560), (1, 2560), 0), out=buf305)
        del arg199_1
        buf306 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf305, arg200_1, buf306, 1310720, grid=grid(1310720), stream=stream0)
        del arg200_1
        buf307 = reinterpret_tensor(buf305, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [contiguous_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf304, arg198_1, buf307, 1310720, grid=grid(1310720), stream=stream0)
        del arg198_1
        buf308 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf306, (128, 80, 128), (10240, 1, 80), 0), out=buf308)
        buf313 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_51], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf308, buf313, 16384, 128, grid=grid(16384), stream=stream0)
        buf311 = reinterpret_tensor(buf307, (512, 2560), (2560, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf303, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg201_1, (2560, 2560), (1, 2560), 0), out=buf311)
        del arg201_1
        buf312 = reinterpret_tensor(buf303, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [value_states_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf311, arg202_1, buf312, 1310720, grid=grid(1310720), stream=stream0)
        del arg202_1
        buf314 = reinterpret_tensor(buf311, (128, 128, 80), (10240, 80, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_51, attn_output_60], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf313, reinterpret_tensor(buf312, (128, 128, 80), (10240, 80, 1), 0), out=buf314)
        buf315 = reinterpret_tensor(buf304, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [attn_output_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf314, buf315, 1310720, grid=grid(1310720), stream=stream0)
        buf316 = reinterpret_tensor(buf314, (512, 2560), (2560, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf315, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg203_1, (2560, 2560), (1, 2560), 0), out=buf316)
        del arg203_1
        buf317 = reinterpret_tensor(buf316, (4, 128, 2560), (327680, 2560, 1), 0); del buf316  # reuse
        buf321 = reinterpret_tensor(buf315, (4, 128, 2560), (327680, 2560, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, hidden_states_112, hidden_states_113], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf317, buf292, buf299, arg194_1, arg204_1, arg205_1, arg206_1, buf321, 512, 2560, grid=grid(512), stream=stream0)
        del arg194_1
        del arg204_1
        del arg205_1
        del arg206_1
        buf322 = reinterpret_tensor(buf298, (512, 10240), (10240, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf321, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg207_1, (2560, 10240), (1, 2560), 0), out=buf322)
        del arg207_1
        buf323 = reinterpret_tensor(buf322, (4, 128, 10240), (1310720, 10240, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_114], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf323, arg208_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg208_1
        buf324 = reinterpret_tensor(buf321, (512, 2560), (2560, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf323, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg209_1, (10240, 2560), (1, 10240), 0), out=buf324)
        del arg209_1
        buf328 = reinterpret_tensor(buf299, (4, 128, 2560), (327680, 2560, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf317, buf324, arg210_1, arg211_1, arg212_1, buf328, 512, 2560, grid=grid(512), stream=stream0)
        del arg211_1
        del arg212_1
        buf329 = reinterpret_tensor(buf292, (512, 2560), (2560, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg213_1, (2560, 2560), (1, 2560), 0), out=buf329)
        del arg213_1
        buf330 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg215_1, (2560, 2560), (1, 2560), 0), out=buf330)
        del arg215_1
        buf331 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf330, arg216_1, buf331, 1310720, grid=grid(1310720), stream=stream0)
        del arg216_1
        buf332 = reinterpret_tensor(buf330, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [contiguous_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf329, arg214_1, buf332, 1310720, grid=grid(1310720), stream=stream0)
        del arg214_1
        buf333 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf331, (128, 80, 128), (10240, 1, 80), 0), out=buf333)
        buf338 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_55], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf333, buf338, 16384, 128, grid=grid(16384), stream=stream0)
        buf336 = reinterpret_tensor(buf332, (512, 2560), (2560, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg217_1, (2560, 2560), (1, 2560), 0), out=buf336)
        del arg217_1
        buf337 = reinterpret_tensor(buf328, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [value_states_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf336, arg218_1, buf337, 1310720, grid=grid(1310720), stream=stream0)
        del arg218_1
        buf339 = reinterpret_tensor(buf336, (128, 128, 80), (10240, 80, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_55, attn_output_65], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf338, reinterpret_tensor(buf337, (128, 128, 80), (10240, 80, 1), 0), out=buf339)
        buf340 = reinterpret_tensor(buf329, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [attn_output_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf339, buf340, 1310720, grid=grid(1310720), stream=stream0)
        buf341 = reinterpret_tensor(buf339, (512, 2560), (2560, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg219_1, (2560, 2560), (1, 2560), 0), out=buf341)
        del arg219_1
        buf342 = reinterpret_tensor(buf341, (4, 128, 2560), (327680, 2560, 1), 0); del buf341  # reuse
        buf346 = reinterpret_tensor(buf340, (4, 128, 2560), (327680, 2560, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_118, hidden_states_121, hidden_states_122], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf342, buf317, buf324, arg210_1, arg220_1, arg221_1, arg222_1, buf346, 512, 2560, grid=grid(512), stream=stream0)
        del arg210_1
        del arg220_1
        del arg221_1
        del arg222_1
        buf347 = reinterpret_tensor(buf323, (512, 10240), (10240, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg223_1, (2560, 10240), (1, 2560), 0), out=buf347)
        del arg223_1
        buf348 = reinterpret_tensor(buf347, (4, 128, 10240), (1310720, 10240, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf348, arg224_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg224_1
        buf349 = reinterpret_tensor(buf346, (512, 2560), (2560, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf348, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg225_1, (10240, 2560), (1, 10240), 0), out=buf349)
        del arg225_1
        buf353 = reinterpret_tensor(buf324, (4, 128, 2560), (327680, 2560, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127, hidden_states_128], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf342, buf349, arg226_1, arg227_1, arg228_1, buf353, 512, 2560, grid=grid(512), stream=stream0)
        del arg227_1
        del arg228_1
        buf354 = reinterpret_tensor(buf317, (512, 2560), (2560, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg229_1, (2560, 2560), (1, 2560), 0), out=buf354)
        del arg229_1
        buf355 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg231_1, (2560, 2560), (1, 2560), 0), out=buf355)
        del arg231_1
        buf356 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf355, arg232_1, buf356, 1310720, grid=grid(1310720), stream=stream0)
        del arg232_1
        buf357 = reinterpret_tensor(buf355, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [contiguous_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf354, arg230_1, buf357, 1310720, grid=grid(1310720), stream=stream0)
        del arg230_1
        buf358 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf357, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf356, (128, 80, 128), (10240, 1, 80), 0), out=buf358)
        buf363 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_59], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf358, buf363, 16384, 128, grid=grid(16384), stream=stream0)
        buf361 = reinterpret_tensor(buf357, (512, 2560), (2560, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg233_1, (2560, 2560), (1, 2560), 0), out=buf361)
        del arg233_1
        buf362 = reinterpret_tensor(buf353, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [value_states_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf361, arg234_1, buf362, 1310720, grid=grid(1310720), stream=stream0)
        del arg234_1
        buf364 = reinterpret_tensor(buf361, (128, 128, 80), (10240, 80, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_59, attn_output_70], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf363, reinterpret_tensor(buf362, (128, 128, 80), (10240, 80, 1), 0), out=buf364)
        buf365 = reinterpret_tensor(buf354, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [attn_output_73], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf364, buf365, 1310720, grid=grid(1310720), stream=stream0)
        buf366 = reinterpret_tensor(buf364, (512, 2560), (2560, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg235_1, (2560, 2560), (1, 2560), 0), out=buf366)
        del arg235_1
        buf367 = reinterpret_tensor(buf366, (4, 128, 2560), (327680, 2560, 1), 0); del buf366  # reuse
        buf371 = reinterpret_tensor(buf365, (4, 128, 2560), (327680, 2560, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127, hidden_states_130, hidden_states_131], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf367, buf342, buf349, arg226_1, arg236_1, arg237_1, arg238_1, buf371, 512, 2560, grid=grid(512), stream=stream0)
        del arg226_1
        del arg236_1
        del arg237_1
        del arg238_1
        buf372 = reinterpret_tensor(buf348, (512, 10240), (10240, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf371, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg239_1, (2560, 10240), (1, 2560), 0), out=buf372)
        del arg239_1
        buf373 = reinterpret_tensor(buf372, (4, 128, 10240), (1310720, 10240, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf373, arg240_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg240_1
        buf374 = reinterpret_tensor(buf371, (512, 2560), (2560, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg241_1, (10240, 2560), (1, 10240), 0), out=buf374)
        del arg241_1
        buf378 = reinterpret_tensor(buf349, (4, 128, 2560), (327680, 2560, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf367, buf374, arg242_1, arg243_1, arg244_1, buf378, 512, 2560, grid=grid(512), stream=stream0)
        del arg243_1
        del arg244_1
        buf379 = reinterpret_tensor(buf342, (512, 2560), (2560, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg245_1, (2560, 2560), (1, 2560), 0), out=buf379)
        del arg245_1
        buf380 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg247_1, (2560, 2560), (1, 2560), 0), out=buf380)
        del arg247_1
        buf381 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf380, arg248_1, buf381, 1310720, grid=grid(1310720), stream=stream0)
        del arg248_1
        buf382 = reinterpret_tensor(buf380, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [contiguous_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf379, arg246_1, buf382, 1310720, grid=grid(1310720), stream=stream0)
        del arg246_1
        buf383 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf382, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf381, (128, 80, 128), (10240, 1, 80), 0), out=buf383)
        buf388 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_63], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf383, buf388, 16384, 128, grid=grid(16384), stream=stream0)
        buf386 = reinterpret_tensor(buf382, (512, 2560), (2560, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg249_1, (2560, 2560), (1, 2560), 0), out=buf386)
        del arg249_1
        buf387 = reinterpret_tensor(buf378, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf378  # reuse
        # Topologically Sorted Source Nodes: [value_states_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf386, arg250_1, buf387, 1310720, grid=grid(1310720), stream=stream0)
        del arg250_1
        buf389 = reinterpret_tensor(buf386, (128, 128, 80), (10240, 80, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_63, attn_output_75], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf388, reinterpret_tensor(buf387, (128, 128, 80), (10240, 80, 1), 0), out=buf389)
        buf390 = reinterpret_tensor(buf379, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [attn_output_78], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf389, buf390, 1310720, grid=grid(1310720), stream=stream0)
        buf391 = reinterpret_tensor(buf389, (512, 2560), (2560, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf390, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg251_1, (2560, 2560), (1, 2560), 0), out=buf391)
        del arg251_1
        buf392 = reinterpret_tensor(buf391, (4, 128, 2560), (327680, 2560, 1), 0); del buf391  # reuse
        buf396 = reinterpret_tensor(buf390, (4, 128, 2560), (327680, 2560, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_139, hidden_states_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf392, buf367, buf374, arg242_1, arg252_1, arg253_1, arg254_1, buf396, 512, 2560, grid=grid(512), stream=stream0)
        del arg242_1
        del arg252_1
        del arg253_1
        del arg254_1
        buf397 = reinterpret_tensor(buf373, (512, 10240), (10240, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf396, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg255_1, (2560, 10240), (1, 2560), 0), out=buf397)
        del arg255_1
        buf398 = reinterpret_tensor(buf397, (4, 128, 10240), (1310720, 10240, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf398, arg256_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg256_1
        buf399 = reinterpret_tensor(buf396, (512, 2560), (2560, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf398, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg257_1, (10240, 2560), (1, 10240), 0), out=buf399)
        del arg257_1
        buf403 = reinterpret_tensor(buf374, (4, 128, 2560), (327680, 2560, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_146], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf392, buf399, arg258_1, arg259_1, arg260_1, buf403, 512, 2560, grid=grid(512), stream=stream0)
        del arg259_1
        del arg260_1
        buf404 = reinterpret_tensor(buf367, (512, 2560), (2560, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg261_1, (2560, 2560), (1, 2560), 0), out=buf404)
        del arg261_1
        buf405 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg263_1, (2560, 2560), (1, 2560), 0), out=buf405)
        del arg263_1
        buf406 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf405, arg264_1, buf406, 1310720, grid=grid(1310720), stream=stream0)
        del arg264_1
        buf407 = reinterpret_tensor(buf405, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [contiguous_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf404, arg262_1, buf407, 1310720, grid=grid(1310720), stream=stream0)
        del arg262_1
        buf408 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf406, (128, 80, 128), (10240, 1, 80), 0), out=buf408)
        buf413 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_67], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf408, buf413, 16384, 128, grid=grid(16384), stream=stream0)
        buf411 = reinterpret_tensor(buf407, (512, 2560), (2560, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg265_1, (2560, 2560), (1, 2560), 0), out=buf411)
        del arg265_1
        buf412 = reinterpret_tensor(buf403, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [value_states_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf411, arg266_1, buf412, 1310720, grid=grid(1310720), stream=stream0)
        del arg266_1
        buf414 = reinterpret_tensor(buf411, (128, 128, 80), (10240, 80, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_67, attn_output_80], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf413, reinterpret_tensor(buf412, (128, 128, 80), (10240, 80, 1), 0), out=buf414)
        buf415 = reinterpret_tensor(buf404, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [attn_output_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf414, buf415, 1310720, grid=grid(1310720), stream=stream0)
        buf416 = reinterpret_tensor(buf414, (512, 2560), (2560, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf415, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg267_1, (2560, 2560), (1, 2560), 0), out=buf416)
        del arg267_1
        buf417 = reinterpret_tensor(buf416, (4, 128, 2560), (327680, 2560, 1), 0); del buf416  # reuse
        buf421 = reinterpret_tensor(buf415, (4, 128, 2560), (327680, 2560, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_145, hidden_states_148, hidden_states_149], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf417, buf392, buf399, arg258_1, arg268_1, arg269_1, arg270_1, buf421, 512, 2560, grid=grid(512), stream=stream0)
        del arg258_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf422 = reinterpret_tensor(buf398, (512, 10240), (10240, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf421, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg271_1, (2560, 10240), (1, 2560), 0), out=buf422)
        del arg271_1
        buf423 = reinterpret_tensor(buf422, (4, 128, 10240), (1310720, 10240, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_150], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf423, arg272_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg272_1
        buf424 = reinterpret_tensor(buf421, (512, 2560), (2560, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg273_1, (10240, 2560), (1, 10240), 0), out=buf424)
        del arg273_1
        buf428 = reinterpret_tensor(buf399, (4, 128, 2560), (327680, 2560, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, hidden_states_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf417, buf424, arg274_1, arg275_1, arg276_1, buf428, 512, 2560, grid=grid(512), stream=stream0)
        del arg275_1
        del arg276_1
        buf429 = reinterpret_tensor(buf392, (512, 2560), (2560, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg277_1, (2560, 2560), (1, 2560), 0), out=buf429)
        del arg277_1
        buf430 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg279_1, (2560, 2560), (1, 2560), 0), out=buf430)
        del arg279_1
        buf431 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf430, arg280_1, buf431, 1310720, grid=grid(1310720), stream=stream0)
        del arg280_1
        buf432 = reinterpret_tensor(buf430, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [contiguous_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf429, arg278_1, buf432, 1310720, grid=grid(1310720), stream=stream0)
        del arg278_1
        buf433 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf431, (128, 80, 128), (10240, 1, 80), 0), out=buf433)
        buf438 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_71], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf433, buf438, 16384, 128, grid=grid(16384), stream=stream0)
        buf436 = reinterpret_tensor(buf432, (512, 2560), (2560, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg281_1, (2560, 2560), (1, 2560), 0), out=buf436)
        del arg281_1
        buf437 = reinterpret_tensor(buf428, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [value_states_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf436, arg282_1, buf437, 1310720, grid=grid(1310720), stream=stream0)
        del arg282_1
        buf439 = reinterpret_tensor(buf436, (128, 128, 80), (10240, 80, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_71, attn_output_85], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf438, reinterpret_tensor(buf437, (128, 128, 80), (10240, 80, 1), 0), out=buf439)
        buf440 = reinterpret_tensor(buf429, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [attn_output_88], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf439, buf440, 1310720, grid=grid(1310720), stream=stream0)
        buf441 = reinterpret_tensor(buf439, (512, 2560), (2560, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg283_1, (2560, 2560), (1, 2560), 0), out=buf441)
        del arg283_1
        buf442 = reinterpret_tensor(buf441, (4, 128, 2560), (327680, 2560, 1), 0); del buf441  # reuse
        buf446 = reinterpret_tensor(buf440, (4, 128, 2560), (327680, 2560, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_154, hidden_states_157, hidden_states_158], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf442, buf417, buf424, arg274_1, arg284_1, arg285_1, arg286_1, buf446, 512, 2560, grid=grid(512), stream=stream0)
        del arg274_1
        del arg284_1
        del arg285_1
        del arg286_1
        buf447 = reinterpret_tensor(buf423, (512, 10240), (10240, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf446, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg287_1, (2560, 10240), (1, 2560), 0), out=buf447)
        del arg287_1
        buf448 = reinterpret_tensor(buf447, (4, 128, 10240), (1310720, 10240, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_159], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf448, arg288_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg288_1
        buf449 = reinterpret_tensor(buf446, (512, 2560), (2560, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf448, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg289_1, (10240, 2560), (1, 10240), 0), out=buf449)
        del arg289_1
        buf453 = reinterpret_tensor(buf424, (4, 128, 2560), (327680, 2560, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_164], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf442, buf449, arg290_1, arg291_1, arg292_1, buf453, 512, 2560, grid=grid(512), stream=stream0)
        del arg291_1
        del arg292_1
        buf454 = reinterpret_tensor(buf417, (512, 2560), (2560, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg293_1, (2560, 2560), (1, 2560), 0), out=buf454)
        del arg293_1
        buf455 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg295_1, (2560, 2560), (1, 2560), 0), out=buf455)
        del arg295_1
        buf456 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf455, arg296_1, buf456, 1310720, grid=grid(1310720), stream=stream0)
        del arg296_1
        buf457 = reinterpret_tensor(buf455, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [contiguous_56], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf454, arg294_1, buf457, 1310720, grid=grid(1310720), stream=stream0)
        del arg294_1
        buf458 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf457, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf456, (128, 80, 128), (10240, 1, 80), 0), out=buf458)
        buf463 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_75], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf458, buf463, 16384, 128, grid=grid(16384), stream=stream0)
        buf461 = reinterpret_tensor(buf457, (512, 2560), (2560, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg297_1, (2560, 2560), (1, 2560), 0), out=buf461)
        del arg297_1
        buf462 = reinterpret_tensor(buf453, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [value_states_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf461, arg298_1, buf462, 1310720, grid=grid(1310720), stream=stream0)
        del arg298_1
        buf464 = reinterpret_tensor(buf461, (128, 128, 80), (10240, 80, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_75, attn_output_90], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf463, reinterpret_tensor(buf462, (128, 128, 80), (10240, 80, 1), 0), out=buf464)
        buf465 = reinterpret_tensor(buf454, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [attn_output_93], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf464, buf465, 1310720, grid=grid(1310720), stream=stream0)
        buf466 = reinterpret_tensor(buf464, (512, 2560), (2560, 1), 0); del buf464  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg299_1, (2560, 2560), (1, 2560), 0), out=buf466)
        del arg299_1
        buf467 = reinterpret_tensor(buf466, (4, 128, 2560), (327680, 2560, 1), 0); del buf466  # reuse
        buf471 = reinterpret_tensor(buf465, (4, 128, 2560), (327680, 2560, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163, hidden_states_166, hidden_states_167], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf467, buf442, buf449, arg290_1, arg300_1, arg301_1, arg302_1, buf471, 512, 2560, grid=grid(512), stream=stream0)
        del arg290_1
        del arg300_1
        del arg301_1
        del arg302_1
        buf472 = reinterpret_tensor(buf448, (512, 10240), (10240, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf471, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg303_1, (2560, 10240), (1, 2560), 0), out=buf472)
        del arg303_1
        buf473 = reinterpret_tensor(buf472, (4, 128, 10240), (1310720, 10240, 1), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_168], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf473, arg304_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg304_1
        buf474 = reinterpret_tensor(buf471, (512, 2560), (2560, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf473, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg305_1, (10240, 2560), (1, 10240), 0), out=buf474)
        del arg305_1
        buf478 = reinterpret_tensor(buf449, (4, 128, 2560), (327680, 2560, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_173], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf467, buf474, arg306_1, arg307_1, arg308_1, buf478, 512, 2560, grid=grid(512), stream=stream0)
        del arg307_1
        del arg308_1
        buf479 = reinterpret_tensor(buf442, (512, 2560), (2560, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg309_1, (2560, 2560), (1, 2560), 0), out=buf479)
        del arg309_1
        buf480 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg311_1, (2560, 2560), (1, 2560), 0), out=buf480)
        del arg311_1
        buf481 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf480, arg312_1, buf481, 1310720, grid=grid(1310720), stream=stream0)
        del arg312_1
        buf482 = reinterpret_tensor(buf480, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf480  # reuse
        # Topologically Sorted Source Nodes: [contiguous_59], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf479, arg310_1, buf482, 1310720, grid=grid(1310720), stream=stream0)
        del arg310_1
        buf483 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_76], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf482, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf481, (128, 80, 128), (10240, 1, 80), 0), out=buf483)
        buf488 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_79], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf483, buf488, 16384, 128, grid=grid(16384), stream=stream0)
        buf486 = reinterpret_tensor(buf482, (512, 2560), (2560, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf478, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg313_1, (2560, 2560), (1, 2560), 0), out=buf486)
        del arg313_1
        buf487 = reinterpret_tensor(buf478, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [value_states_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf486, arg314_1, buf487, 1310720, grid=grid(1310720), stream=stream0)
        del arg314_1
        buf489 = reinterpret_tensor(buf486, (128, 128, 80), (10240, 80, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_79, attn_output_95], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf488, reinterpret_tensor(buf487, (128, 128, 80), (10240, 80, 1), 0), out=buf489)
        buf490 = reinterpret_tensor(buf479, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [attn_output_98], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf489, buf490, 1310720, grid=grid(1310720), stream=stream0)
        buf491 = reinterpret_tensor(buf489, (512, 2560), (2560, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg315_1, (2560, 2560), (1, 2560), 0), out=buf491)
        del arg315_1
        buf492 = reinterpret_tensor(buf491, (4, 128, 2560), (327680, 2560, 1), 0); del buf491  # reuse
        buf496 = reinterpret_tensor(buf490, (4, 128, 2560), (327680, 2560, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_175, hidden_states_176], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf492, buf467, buf474, arg306_1, arg316_1, arg317_1, arg318_1, buf496, 512, 2560, grid=grid(512), stream=stream0)
        del arg306_1
        del arg316_1
        del arg317_1
        del arg318_1
        buf497 = reinterpret_tensor(buf473, (512, 10240), (10240, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg319_1, (2560, 10240), (1, 2560), 0), out=buf497)
        del arg319_1
        buf498 = reinterpret_tensor(buf497, (4, 128, 10240), (1310720, 10240, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf498, arg320_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg320_1
        buf499 = reinterpret_tensor(buf496, (512, 2560), (2560, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg321_1, (10240, 2560), (1, 10240), 0), out=buf499)
        del arg321_1
        buf503 = reinterpret_tensor(buf474, (4, 128, 2560), (327680, 2560, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_181, hidden_states_182], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf492, buf499, arg322_1, arg323_1, arg324_1, buf503, 512, 2560, grid=grid(512), stream=stream0)
        del arg323_1
        del arg324_1
        buf504 = reinterpret_tensor(buf467, (512, 2560), (2560, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg325_1, (2560, 2560), (1, 2560), 0), out=buf504)
        del arg325_1
        buf505 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg327_1, (2560, 2560), (1, 2560), 0), out=buf505)
        del arg327_1
        buf506 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf505, arg328_1, buf506, 1310720, grid=grid(1310720), stream=stream0)
        del arg328_1
        buf507 = reinterpret_tensor(buf505, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [contiguous_62], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf504, arg326_1, buf507, 1310720, grid=grid(1310720), stream=stream0)
        del arg326_1
        buf508 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf507, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf506, (128, 80, 128), (10240, 1, 80), 0), out=buf508)
        buf513 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_83], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf508, buf513, 16384, 128, grid=grid(16384), stream=stream0)
        buf511 = reinterpret_tensor(buf507, (512, 2560), (2560, 1), 0); del buf507  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf503, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg329_1, (2560, 2560), (1, 2560), 0), out=buf511)
        del arg329_1
        buf512 = reinterpret_tensor(buf503, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [value_states_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf511, arg330_1, buf512, 1310720, grid=grid(1310720), stream=stream0)
        del arg330_1
        buf514 = reinterpret_tensor(buf511, (128, 128, 80), (10240, 80, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_83, attn_output_100], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf513, reinterpret_tensor(buf512, (128, 128, 80), (10240, 80, 1), 0), out=buf514)
        buf515 = reinterpret_tensor(buf504, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [attn_output_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf514, buf515, 1310720, grid=grid(1310720), stream=stream0)
        buf516 = reinterpret_tensor(buf514, (512, 2560), (2560, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg331_1, (2560, 2560), (1, 2560), 0), out=buf516)
        del arg331_1
        buf517 = reinterpret_tensor(buf516, (4, 128, 2560), (327680, 2560, 1), 0); del buf516  # reuse
        buf521 = reinterpret_tensor(buf515, (4, 128, 2560), (327680, 2560, 1), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_181, hidden_states_184, hidden_states_185], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf517, buf492, buf499, arg322_1, arg332_1, arg333_1, arg334_1, buf521, 512, 2560, grid=grid(512), stream=stream0)
        del arg322_1
        del arg332_1
        del arg333_1
        del arg334_1
        buf522 = reinterpret_tensor(buf498, (512, 10240), (10240, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf521, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg335_1, (2560, 10240), (1, 2560), 0), out=buf522)
        del arg335_1
        buf523 = reinterpret_tensor(buf522, (4, 128, 10240), (1310720, 10240, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_186], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf523, arg336_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg336_1
        buf524 = reinterpret_tensor(buf521, (512, 2560), (2560, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf523, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg337_1, (10240, 2560), (1, 10240), 0), out=buf524)
        del arg337_1
        buf528 = reinterpret_tensor(buf499, (4, 128, 2560), (327680, 2560, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf517, buf524, arg338_1, arg339_1, arg340_1, buf528, 512, 2560, grid=grid(512), stream=stream0)
        del arg339_1
        del arg340_1
        buf529 = reinterpret_tensor(buf492, (512, 2560), (2560, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg341_1, (2560, 2560), (1, 2560), 0), out=buf529)
        del arg341_1
        buf530 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg343_1, (2560, 2560), (1, 2560), 0), out=buf530)
        del arg343_1
        buf531 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf530, arg344_1, buf531, 1310720, grid=grid(1310720), stream=stream0)
        del arg344_1
        buf532 = reinterpret_tensor(buf530, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [contiguous_65], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf529, arg342_1, buf532, 1310720, grid=grid(1310720), stream=stream0)
        del arg342_1
        buf533 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf531, (128, 80, 128), (10240, 1, 80), 0), out=buf533)
        buf538 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_87], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf533, buf538, 16384, 128, grid=grid(16384), stream=stream0)
        buf536 = reinterpret_tensor(buf532, (512, 2560), (2560, 1), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf528, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg345_1, (2560, 2560), (1, 2560), 0), out=buf536)
        del arg345_1
        buf537 = reinterpret_tensor(buf528, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [value_states_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf536, arg346_1, buf537, 1310720, grid=grid(1310720), stream=stream0)
        del arg346_1
        buf539 = reinterpret_tensor(buf536, (128, 128, 80), (10240, 80, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_87, attn_output_105], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf538, reinterpret_tensor(buf537, (128, 128, 80), (10240, 80, 1), 0), out=buf539)
        buf540 = reinterpret_tensor(buf529, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [attn_output_108], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf539, buf540, 1310720, grid=grid(1310720), stream=stream0)
        buf541 = reinterpret_tensor(buf539, (512, 2560), (2560, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf540, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg347_1, (2560, 2560), (1, 2560), 0), out=buf541)
        del arg347_1
        buf542 = reinterpret_tensor(buf541, (4, 128, 2560), (327680, 2560, 1), 0); del buf541  # reuse
        buf546 = reinterpret_tensor(buf540, (4, 128, 2560), (327680, 2560, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_193, hidden_states_194], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf542, buf517, buf524, arg338_1, arg348_1, arg349_1, arg350_1, buf546, 512, 2560, grid=grid(512), stream=stream0)
        del arg338_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf547 = reinterpret_tensor(buf523, (512, 10240), (10240, 1), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg351_1, (2560, 10240), (1, 2560), 0), out=buf547)
        del arg351_1
        buf548 = reinterpret_tensor(buf547, (4, 128, 10240), (1310720, 10240, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_195], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf548, arg352_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg352_1
        buf549 = reinterpret_tensor(buf546, (512, 2560), (2560, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg353_1, (10240, 2560), (1, 10240), 0), out=buf549)
        del arg353_1
        buf553 = reinterpret_tensor(buf524, (4, 128, 2560), (327680, 2560, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_200], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf542, buf549, arg354_1, arg355_1, arg356_1, buf553, 512, 2560, grid=grid(512), stream=stream0)
        del arg355_1
        del arg356_1
        buf554 = reinterpret_tensor(buf517, (512, 2560), (2560, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg357_1, (2560, 2560), (1, 2560), 0), out=buf554)
        del arg357_1
        buf555 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg359_1, (2560, 2560), (1, 2560), 0), out=buf555)
        del arg359_1
        buf556 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf555, arg360_1, buf556, 1310720, grid=grid(1310720), stream=stream0)
        del arg360_1
        buf557 = reinterpret_tensor(buf555, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [contiguous_68], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf554, arg358_1, buf557, 1310720, grid=grid(1310720), stream=stream0)
        del arg358_1
        buf558 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_88], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf556, (128, 80, 128), (10240, 1, 80), 0), out=buf558)
        buf563 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_91], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf558, buf563, 16384, 128, grid=grid(16384), stream=stream0)
        buf561 = reinterpret_tensor(buf557, (512, 2560), (2560, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf553, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg361_1, (2560, 2560), (1, 2560), 0), out=buf561)
        del arg361_1
        buf562 = reinterpret_tensor(buf553, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [value_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf561, arg362_1, buf562, 1310720, grid=grid(1310720), stream=stream0)
        del arg362_1
        buf564 = reinterpret_tensor(buf561, (128, 128, 80), (10240, 80, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_91, attn_output_110], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf563, reinterpret_tensor(buf562, (128, 128, 80), (10240, 80, 1), 0), out=buf564)
        buf565 = reinterpret_tensor(buf554, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [attn_output_113], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf564, buf565, 1310720, grid=grid(1310720), stream=stream0)
        buf566 = reinterpret_tensor(buf564, (512, 2560), (2560, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf565, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg363_1, (2560, 2560), (1, 2560), 0), out=buf566)
        del arg363_1
        buf567 = reinterpret_tensor(buf566, (4, 128, 2560), (327680, 2560, 1), 0); del buf566  # reuse
        buf571 = reinterpret_tensor(buf565, (4, 128, 2560), (327680, 2560, 1), 0); del buf565  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_199, hidden_states_202, hidden_states_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf567, buf542, buf549, arg354_1, arg364_1, arg365_1, arg366_1, buf571, 512, 2560, grid=grid(512), stream=stream0)
        del arg354_1
        del arg364_1
        del arg365_1
        del arg366_1
        buf572 = reinterpret_tensor(buf548, (512, 10240), (10240, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf571, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg367_1, (2560, 10240), (1, 2560), 0), out=buf572)
        del arg367_1
        buf573 = reinterpret_tensor(buf572, (4, 128, 10240), (1310720, 10240, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_204], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf573, arg368_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg368_1
        buf574 = reinterpret_tensor(buf571, (512, 2560), (2560, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg369_1, (10240, 2560), (1, 10240), 0), out=buf574)
        del arg369_1
        buf578 = reinterpret_tensor(buf549, (4, 128, 2560), (327680, 2560, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_208, hidden_states_209], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf567, buf574, arg370_1, arg371_1, arg372_1, buf578, 512, 2560, grid=grid(512), stream=stream0)
        del arg371_1
        del arg372_1
        buf579 = reinterpret_tensor(buf542, (512, 2560), (2560, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg373_1, (2560, 2560), (1, 2560), 0), out=buf579)
        del arg373_1
        buf580 = empty_strided_cuda((512, 2560), (2560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg375_1, (2560, 2560), (1, 2560), 0), out=buf580)
        del arg375_1
        buf581 = empty_strided_cuda((4, 32, 128, 80), (327680, 10240, 80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf580, arg376_1, buf581, 1310720, grid=grid(1310720), stream=stream0)
        del arg376_1
        buf582 = reinterpret_tensor(buf580, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [contiguous_71], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf579, arg374_1, buf582, 1310720, grid=grid(1310720), stream=stream0)
        del arg374_1
        buf583 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_92], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (128, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf581, (128, 80, 128), (10240, 1, 80), 0), out=buf583)
        buf588 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_95], Original ATen: [aten._softmax]
        triton_per_fused__softmax_3.run(buf583, buf588, 16384, 128, grid=grid(16384), stream=stream0)
        del buf583
        buf586 = reinterpret_tensor(buf582, (512, 2560), (2560, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf578, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg377_1, (2560, 2560), (1, 2560), 0), out=buf586)
        del arg377_1
        buf587 = reinterpret_tensor(buf578, (4, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [value_states_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_1.run(buf586, arg378_1, buf587, 1310720, grid=grid(1310720), stream=stream0)
        del arg378_1
        buf589 = reinterpret_tensor(buf586, (128, 128, 80), (10240, 80, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_95, attn_output_115], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf588, reinterpret_tensor(buf587, (128, 128, 80), (10240, 80, 1), 0), out=buf589)
        del buf588
        buf590 = reinterpret_tensor(buf579, (4, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [attn_output_118], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf589, buf590, 1310720, grid=grid(1310720), stream=stream0)
        buf591 = reinterpret_tensor(buf589, (512, 2560), (2560, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf590, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg379_1, (2560, 2560), (1, 2560), 0), out=buf591)
        del arg379_1
        buf592 = reinterpret_tensor(buf591, (4, 128, 2560), (327680, 2560, 1), 0); del buf591  # reuse
        buf596 = reinterpret_tensor(buf590, (4, 128, 2560), (327680, 2560, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_208, hidden_states_211, hidden_states_212], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_8.run(buf592, buf567, buf574, arg370_1, arg380_1, arg381_1, arg382_1, buf596, 512, 2560, grid=grid(512), stream=stream0)
        del arg370_1
        del arg380_1
        del arg381_1
        del arg382_1
        del buf567
        buf597 = reinterpret_tensor(buf573, (512, 10240), (10240, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg383_1, (2560, 10240), (1, 2560), 0), out=buf597)
        del arg383_1
        buf598 = reinterpret_tensor(buf597, (4, 128, 10240), (1310720, 10240, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_213], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_6.run(buf598, arg384_1, 5242880, grid=grid(5242880), stream=stream0)
        del arg384_1
        buf599 = reinterpret_tensor(buf596, (512, 2560), (2560, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (512, 10240), (10240, 1), 0), reinterpret_tensor(arg385_1, (10240, 2560), (1, 10240), 0), out=buf599)
        del arg385_1
        del buf598
        buf603 = reinterpret_tensor(buf574, (4, 128, 2560), (327680, 2560, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_217, hidden_states_218], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_7.run(buf592, buf599, arg386_1, arg387_1, arg388_1, buf603, 512, 2560, grid=grid(512), stream=stream0)
        del arg386_1
        del arg387_1
        del arg388_1
        del buf592
        del buf599
        buf604 = empty_strided_cuda((512, 8008), (8032, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (512, 2560), (2560, 1), 0), reinterpret_tensor(arg1_1, (2560, 8008), (1, 2560), 0), out=buf604)
        del arg1_1
        del buf603
        buf605 = empty_strided_cuda((512, 1), (1, 512), torch.float32)
        buf606 = empty_strided_cuda((512, 1), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_9.run(buf604, buf605, buf606, 512, 8008, grid=grid(512), stream=stream0)
        buf607 = empty_strided_cuda((), (), torch.float32)
        buf609 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_10.run(buf609, arg389_1, buf604, buf605, buf606, 1, 512, grid=grid(1), stream=stream0)
        del arg389_1
        del buf605
        del buf606
    return (buf609, reinterpret_tensor(buf604, (4, 128, 8008), (1028096, 8032, 1), 0), buf6, buf12, buf31, buf37, buf56, buf62, buf81, buf87, buf106, buf112, buf131, buf137, buf156, buf162, buf181, buf187, buf206, buf212, buf231, buf237, buf256, buf262, buf281, buf287, buf306, buf312, buf331, buf337, buf356, buf362, buf381, buf387, buf406, buf412, buf431, buf437, buf456, buf462, buf481, buf487, buf506, buf512, buf531, buf537, buf556, buf562, buf581, buf587, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((8008, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((10240, 2560), (2560, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((10240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((2560, 10240), (10240, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotForCausalLM', benchmark_compiled_module)
