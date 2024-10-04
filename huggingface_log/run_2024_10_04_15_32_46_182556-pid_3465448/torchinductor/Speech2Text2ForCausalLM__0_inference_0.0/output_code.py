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


# kernel path: /tmp/torchinductor_sahanp/lu/clumu5xahowwl5cqccirqxqrsxx6episl5lqunrjfoc4rxvgq7ow.py
# Topologically Sorted Source Nodes: [ne, mask_2, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   mask_2 => convert_element_type
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
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_cumsum_ne_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_sahanp/kk/ckkbcyyhzvr3jcibs24bn2zddso23esp5roae6mo76lz4sqt4w32.py
# Topologically Sorted Source Nodes: [embedding, inputs_embeds, hidden_states], Original ATen: [aten.embedding, aten.mul, aten.add]
# Source node to ATen node mapping:
#   embedding => embedding
#   hidden_states => add_3
#   inputs_embeds => mul
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %view, 1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, 16.0), kwargs = {})
#   %add_3 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %view_3), kwargs = {})
triton_poi_fused_add_embedding_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 10000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 10000), "index out of bounds: 0 <= tmp4 < 10000")
    tmp6 = tl.load(in_ptr1 + (x0 + (256*tmp4)), None)
    tmp7 = 16.0
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
    tmp19 = tl.full([XBLOCK], 1026, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 1026), "index out of bounds: 0 <= tmp22 < 1026")
    tmp24 = tl.load(in_ptr3 + (x0 + (256*tmp22)), None)
    tmp25 = tmp8 + tmp24
    tl.store(out_ptr0 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gq/cgqqlp4kpvvzl5n4jpsa7guposowqzdhbw7lcz6q2gbbhom2kkv2.py
# Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_states => clone_1
# Graph fragment:
#   %clone_1 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 4
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (32768*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iw/ciwykyaazihv4idgp67e2fy2y7x3axefdrvc45t3b6727q6x4pmw.py
# Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_2 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    x2 = (xindex // 8192) % 4
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (32768*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/by/cbygbovdmuodnlfhorgjivpytldscub63yzfbjdhdh6mkteeezhf.py
# Topologically Sorted Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_weights_3 => amax, div, exp, sub, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_17, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_17, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_sahanp/h3/ch3x7s72nc4nfmzray3rwqxmhu2jmgkobal7k6qxhwzlsnsyn4ks.py
# Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_3 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x2 = (xindex // 256) % 128
    x3 = (xindex // 32768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (8192*x1) + (32768*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3u/c3uk4ry6zazjxuso2oaexz2eilh5edc2rzbvxv2wdwwxglsw3vcp.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_3 => add_5
#   hidden_states_4 => add_6, add_7, mul_3, mul_4, rsqrt, sub_1, var_mean
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_21), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg11_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg12_1), kwargs = {})
triton_per_fused_add_native_layer_norm_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 32768
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3c/c3c7vjxnpql4lla4thi4cdnksfldnu67qlhdpx4e6r5docogvz6d.py
# Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   hidden_states_5 => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_23,), kwargs = {})
triton_poi_fused_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4u/c4ucowdspu4cdxyz7ba2lkcwgruw5eoi5yddeoparbuohvxuhwpg.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_6, exp_6, sub_18, sum_7
# Graph fragment:
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_138, [1], True), kwargs = {})
#   %sub_18 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_138, %amax_6), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_18,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [1], True), kwargs = {})
triton_red_fused__log_softmax_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 16384],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 10000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (10016*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (10016*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/km/ckmojvtvfedc3zxp4t3myfkqdylrf22zioxba3fb4sr7vsepozeo.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => full_default_3, ne_2, ne_3, neg, sum_8, sum_9, where_2
# Graph fragment:
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_139, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_3), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_139, -100), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_3,), kwargs = {})
triton_red_fused_nll_loss_forward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 10000, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 10000)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp8 < 10000")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (10016*r1) + (82051072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_sahanp/g5/cg5jx7llhodtgimpudhfzudth7a2wq6jcxdztywd2qebajfjjcjh.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_3, div_6, full_default_3, ne_2, ne_3, neg, sum_8, sum_9, where_2
# Graph fragment:
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_139, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_3), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %ne_3 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_139, -100), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_3,), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_8, torch.float32), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_9, %convert_element_type_3), kwargs = {})
triton_per_fused_nll_loss_forward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1 = args
    args.clear()
    assert_size_stride(arg0_1, (256, 128), (128, 1))
    assert_size_stride(arg1_1, (10000, 256), (256, 1))
    assert_size_stride(arg2_1, (1026, 256), (256, 1))
    assert_size_stride(arg3_1, (256, 256), (256, 1))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg5_1, (256, 256), (256, 1))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (256, 256), (256, 1))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 256), (256, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (2048, 256), (256, 1))
    assert_size_stride(arg14_1, (2048, ), (1, ))
    assert_size_stride(arg15_1, (256, 2048), (2048, 1))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, 256), (256, 1))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, 256), (256, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, 256), (256, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, 256), (256, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (2048, 256), (256, 1))
    assert_size_stride(arg30_1, (2048, ), (1, ))
    assert_size_stride(arg31_1, (256, 2048), (2048, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, 256), (256, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 256), (256, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 256), (256, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256), (256, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (2048, 256), (256, 1))
    assert_size_stride(arg46_1, (2048, ), (1, ))
    assert_size_stride(arg47_1, (256, 2048), (2048, 1))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, 256), (256, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, 256), (256, 1))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, 256), (256, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, 256), (256, 1))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (2048, 256), (256, 1))
    assert_size_stride(arg62_1, (2048, ), (1, ))
    assert_size_stride(arg63_1, (256, 2048), (2048, 1))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, 256), (256, 1))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, 256), (256, 1))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, 256), (256, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, 256), (256, 1))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg77_1, (2048, 256), (256, 1))
    assert_size_stride(arg78_1, (2048, ), (1, ))
    assert_size_stride(arg79_1, (256, 2048), (2048, 1))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, 256), (256, 1))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, 256), (256, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, 256), (256, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256), (256, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (2048, 256), (256, 1))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg95_1, (256, 2048), (2048, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((256, 128), (128, 1), torch.int64)
        # Topologically Sorted Source Nodes: [ne, mask_2, cumsum], Original ATen: [aten.ne, aten._to_copy, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_cumsum_ne_0.run(arg0_1, buf0, 256, 128, grid=grid(256), stream=stream0)
        buf1 = empty_strided_cuda((256, 128, 256), (32768, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, inputs_embeds, hidden_states], Original ATen: [aten.embedding, aten.mul, aten.add]
        triton_poi_fused_add_embedding_mul_1.run(arg0_1, arg1_1, buf0, arg2_1, buf1, 8388608, grid=grid(8388608), stream=stream0)
        del arg0_1
        del arg2_1
        del buf0
        buf2 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1, (32768, 256), (256, 1), 0), reinterpret_tensor(arg3_1, (256, 256), (1, 256), 0), out=buf2)
        del arg3_1
        buf3 = empty_strided_cuda((32768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1, (32768, 256), (256, 1), 0), reinterpret_tensor(arg5_1, (256, 256), (1, 256), 0), out=buf3)
        del arg5_1
        buf4 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf3, arg6_1, buf4, 8388608, grid=grid(8388608), stream=stream0)
        del arg6_1
        buf5 = reinterpret_tensor(buf3, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf2, arg4_1, buf5, 8388608, grid=grid(8388608), stream=stream0)
        del arg4_1
        buf6 = empty_strided_cuda((1024, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf4, (1024, 64, 128), (8192, 1, 64), 0), out=buf6)
        buf11 = empty_strided_cuda((1024, 128, 128), (16384, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_weights_3], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf6, buf11, 131072, 128, grid=grid(131072), stream=stream0)
        buf9 = reinterpret_tensor(buf5, (32768, 256), (256, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1, (32768, 256), (256, 1), 0), reinterpret_tensor(arg7_1, (256, 256), (1, 256), 0), out=buf9)
        del arg7_1
        buf10 = reinterpret_tensor(buf2, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [value_states], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf9, arg8_1, buf10, 8388608, grid=grid(8388608), stream=stream0)
        del arg8_1
        buf12 = reinterpret_tensor(buf9, (1024, 128, 64), (8192, 64, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_3, attn_output], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf11, reinterpret_tensor(buf10, (1024, 128, 64), (8192, 64, 1), 0), out=buf12)
        buf13 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf12, buf13, 8388608, grid=grid(8388608), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (32768, 256), (256, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf13, (32768, 256), (256, 1), 0), reinterpret_tensor(arg9_1, (256, 256), (1, 256), 0), out=buf14)
        del arg9_1
        buf18 = reinterpret_tensor(buf13, (256, 128, 256), (32768, 256, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf1, buf14, arg10_1, arg11_1, arg12_1, buf18, 32768, 256, grid=grid(32768), stream=stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        buf19 = empty_strided_cuda((32768, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf18, (32768, 256), (256, 1), 0), reinterpret_tensor(arg13_1, (256, 2048), (1, 256), 0), out=buf19)
        del arg13_1
        buf20 = reinterpret_tensor(buf19, (256, 128, 2048), (262144, 2048, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf20, arg14_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg14_1
        buf21 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf20, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg15_1, (2048, 256), (1, 2048), 0), out=buf21)
        del arg15_1
        buf25 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9, hidden_states_10], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf18, buf21, arg16_1, arg17_1, arg18_1, buf25, 32768, 256, grid=grid(32768), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg19_1, (256, 256), (1, 256), 0), out=buf26)
        del arg19_1
        buf27 = reinterpret_tensor(buf18, (32768, 256), (256, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg21_1, (256, 256), (1, 256), 0), out=buf27)
        del arg21_1
        buf28 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf27, arg22_1, buf28, 8388608, grid=grid(8388608), stream=stream0)
        del arg22_1
        buf29 = reinterpret_tensor(buf27, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf26, arg20_1, buf29, 8388608, grid=grid(8388608), stream=stream0)
        del arg20_1
        buf30 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf28, (1024, 64, 128), (8192, 1, 64), 0), out=buf30)
        buf35 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf30, buf35, 131072, 128, grid=grid(131072), stream=stream0)
        buf33 = reinterpret_tensor(buf29, (32768, 256), (256, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (32768, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 256), (1, 256), 0), out=buf33)
        del arg23_1
        buf34 = reinterpret_tensor(buf26, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [value_states_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf33, arg24_1, buf34, 8388608, grid=grid(8388608), stream=stream0)
        del arg24_1
        buf36 = reinterpret_tensor(buf33, (1024, 128, 64), (8192, 64, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_7, attn_output_5], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf35, reinterpret_tensor(buf34, (1024, 128, 64), (8192, 64, 1), 0), out=buf36)
        buf37 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf36, buf37, 8388608, grid=grid(8388608), stream=stream0)
        buf38 = reinterpret_tensor(buf36, (32768, 256), (256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (32768, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 256), (1, 256), 0), out=buf38)
        del arg25_1
        buf42 = reinterpret_tensor(buf37, (256, 128, 256), (32768, 256, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12, hidden_states_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf25, buf38, arg26_1, arg27_1, arg28_1, buf42, 32768, 256, grid=grid(32768), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        buf43 = reinterpret_tensor(buf20, (32768, 2048), (2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf42, (32768, 256), (256, 1), 0), reinterpret_tensor(arg29_1, (256, 2048), (1, 256), 0), out=buf43)
        del arg29_1
        buf44 = reinterpret_tensor(buf43, (256, 128, 2048), (262144, 2048, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_14], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf44, arg30_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg30_1
        buf45 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf44, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg31_1, (2048, 256), (1, 2048), 0), out=buf45)
        del arg31_1
        buf49 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf42, buf45, arg32_1, arg33_1, arg34_1, buf49, 32768, 256, grid=grid(32768), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf50 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 256), (1, 256), 0), out=buf50)
        del arg35_1
        buf51 = reinterpret_tensor(buf42, (32768, 256), (256, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), reinterpret_tensor(arg37_1, (256, 256), (1, 256), 0), out=buf51)
        del arg37_1
        buf52 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf51, arg38_1, buf52, 8388608, grid=grid(8388608), stream=stream0)
        del arg38_1
        buf53 = reinterpret_tensor(buf51, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf50, arg36_1, buf53, 8388608, grid=grid(8388608), stream=stream0)
        del arg36_1
        buf54 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf52, (1024, 64, 128), (8192, 1, 64), 0), out=buf54)
        buf59 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf54, buf59, 131072, 128, grid=grid(131072), stream=stream0)
        buf57 = reinterpret_tensor(buf53, (32768, 256), (256, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), reinterpret_tensor(arg39_1, (256, 256), (1, 256), 0), out=buf57)
        del arg39_1
        buf58 = reinterpret_tensor(buf50, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [value_states_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf57, arg40_1, buf58, 8388608, grid=grid(8388608), stream=stream0)
        del arg40_1
        buf60 = reinterpret_tensor(buf57, (1024, 128, 64), (8192, 64, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_11, attn_output_10], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf59, reinterpret_tensor(buf58, (1024, 128, 64), (8192, 64, 1), 0), out=buf60)
        buf61 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf60, buf61, 8388608, grid=grid(8388608), stream=stream0)
        buf62 = reinterpret_tensor(buf60, (32768, 256), (256, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf61, (32768, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 256), (1, 256), 0), out=buf62)
        del arg41_1
        buf66 = reinterpret_tensor(buf61, (256, 128, 256), (32768, 256, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf49, buf62, arg42_1, arg43_1, arg44_1, buf66, 32768, 256, grid=grid(32768), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        buf67 = reinterpret_tensor(buf44, (32768, 2048), (2048, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (32768, 256), (256, 1), 0), reinterpret_tensor(arg45_1, (256, 2048), (1, 256), 0), out=buf67)
        del arg45_1
        buf68 = reinterpret_tensor(buf67, (256, 128, 2048), (262144, 2048, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf68, arg46_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg46_1
        buf69 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg47_1, (2048, 256), (1, 2048), 0), out=buf69)
        del arg47_1
        buf73 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf66, buf69, arg48_1, arg49_1, arg50_1, buf73, 32768, 256, grid=grid(32768), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        buf74 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (32768, 256), (256, 1), 0), reinterpret_tensor(arg51_1, (256, 256), (1, 256), 0), out=buf74)
        del arg51_1
        buf75 = reinterpret_tensor(buf66, (32768, 256), (256, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (32768, 256), (256, 1), 0), reinterpret_tensor(arg53_1, (256, 256), (1, 256), 0), out=buf75)
        del arg53_1
        buf76 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf75, arg54_1, buf76, 8388608, grid=grid(8388608), stream=stream0)
        del arg54_1
        buf77 = reinterpret_tensor(buf75, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf74, arg52_1, buf77, 8388608, grid=grid(8388608), stream=stream0)
        del arg52_1
        buf78 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf76, (1024, 64, 128), (8192, 1, 64), 0), out=buf78)
        buf83 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf78, buf83, 131072, 128, grid=grid(131072), stream=stream0)
        buf81 = reinterpret_tensor(buf77, (32768, 256), (256, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf73, (32768, 256), (256, 1), 0), reinterpret_tensor(arg55_1, (256, 256), (1, 256), 0), out=buf81)
        del arg55_1
        buf82 = reinterpret_tensor(buf74, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [value_states_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf81, arg56_1, buf82, 8388608, grid=grid(8388608), stream=stream0)
        del arg56_1
        buf84 = reinterpret_tensor(buf81, (1024, 128, 64), (8192, 64, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_15, attn_output_15], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf83, reinterpret_tensor(buf82, (1024, 128, 64), (8192, 64, 1), 0), out=buf84)
        buf85 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf84, buf85, 8388608, grid=grid(8388608), stream=stream0)
        buf86 = reinterpret_tensor(buf84, (32768, 256), (256, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (32768, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 256), (1, 256), 0), out=buf86)
        del arg57_1
        buf90 = reinterpret_tensor(buf85, (256, 128, 256), (32768, 256, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf73, buf86, arg58_1, arg59_1, arg60_1, buf90, 32768, 256, grid=grid(32768), stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        buf91 = reinterpret_tensor(buf68, (32768, 2048), (2048, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (32768, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 2048), (1, 256), 0), out=buf91)
        del arg61_1
        buf92 = reinterpret_tensor(buf91, (256, 128, 2048), (262144, 2048, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf92, arg62_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg62_1
        buf93 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf92, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 256), (1, 2048), 0), out=buf93)
        del arg63_1
        buf97 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf90, buf93, arg64_1, arg65_1, arg66_1, buf97, 32768, 256, grid=grid(32768), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        buf98 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (32768, 256), (256, 1), 0), reinterpret_tensor(arg67_1, (256, 256), (1, 256), 0), out=buf98)
        del arg67_1
        buf99 = reinterpret_tensor(buf90, (32768, 256), (256, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (32768, 256), (256, 1), 0), reinterpret_tensor(arg69_1, (256, 256), (1, 256), 0), out=buf99)
        del arg69_1
        buf100 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf99, arg70_1, buf100, 8388608, grid=grid(8388608), stream=stream0)
        del arg70_1
        buf101 = reinterpret_tensor(buf99, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [contiguous_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf98, arg68_1, buf101, 8388608, grid=grid(8388608), stream=stream0)
        del arg68_1
        buf102 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf100, (1024, 64, 128), (8192, 1, 64), 0), out=buf102)
        buf107 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf102, buf107, 131072, 128, grid=grid(131072), stream=stream0)
        buf105 = reinterpret_tensor(buf101, (32768, 256), (256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf97, (32768, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), out=buf105)
        del arg71_1
        buf106 = reinterpret_tensor(buf98, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [value_states_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf105, arg72_1, buf106, 8388608, grid=grid(8388608), stream=stream0)
        del arg72_1
        buf108 = reinterpret_tensor(buf105, (1024, 128, 64), (8192, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_19, attn_output_20], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf107, reinterpret_tensor(buf106, (1024, 128, 64), (8192, 64, 1), 0), out=buf108)
        buf109 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf108, buf109, 8388608, grid=grid(8388608), stream=stream0)
        buf110 = reinterpret_tensor(buf108, (32768, 256), (256, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf109, (32768, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 256), (1, 256), 0), out=buf110)
        del arg73_1
        buf114 = reinterpret_tensor(buf109, (256, 128, 256), (32768, 256, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf97, buf110, arg74_1, arg75_1, arg76_1, buf114, 32768, 256, grid=grid(32768), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        buf115 = reinterpret_tensor(buf92, (32768, 2048), (2048, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (32768, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 2048), (1, 256), 0), out=buf115)
        del arg77_1
        buf116 = reinterpret_tensor(buf115, (256, 128, 2048), (262144, 2048, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf116, arg78_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg78_1
        buf117 = reinterpret_tensor(buf97, (32768, 256), (256, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg79_1, (2048, 256), (1, 2048), 0), out=buf117)
        del arg79_1
        buf121 = reinterpret_tensor(buf110, (256, 128, 256), (32768, 256, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45, hidden_states_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf114, buf117, arg80_1, arg81_1, arg82_1, buf121, 32768, 256, grid=grid(32768), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        buf122 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (32768, 256), (256, 1), 0), reinterpret_tensor(arg83_1, (256, 256), (1, 256), 0), out=buf122)
        del arg83_1
        buf123 = reinterpret_tensor(buf114, (32768, 256), (256, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (32768, 256), (256, 1), 0), reinterpret_tensor(arg85_1, (256, 256), (1, 256), 0), out=buf123)
        del arg85_1
        buf124 = empty_strided_cuda((256, 4, 128, 64), (32768, 8192, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf123, arg86_1, buf124, 8388608, grid=grid(8388608), stream=stream0)
        del arg86_1
        buf125 = reinterpret_tensor(buf123, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf122, arg84_1, buf125, 8388608, grid=grid(8388608), stream=stream0)
        del arg84_1
        buf126 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (1024, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf124, (1024, 64, 128), (8192, 1, 64), 0), out=buf126)
        buf131 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23], Original ATen: [aten._softmax]
        triton_per_fused__softmax_4.run(buf126, buf131, 131072, 128, grid=grid(131072), stream=stream0)
        del buf126
        buf129 = reinterpret_tensor(buf125, (32768, 256), (256, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (32768, 256), (256, 1), 0), reinterpret_tensor(arg87_1, (256, 256), (1, 256), 0), out=buf129)
        del arg87_1
        buf130 = reinterpret_tensor(buf122, (256, 4, 128, 64), (32768, 8192, 64, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [value_states_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf129, arg88_1, buf130, 8388608, grid=grid(8388608), stream=stream0)
        del arg88_1
        buf132 = reinterpret_tensor(buf129, (1024, 128, 64), (8192, 64, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_23, attn_output_25], Original ATen: [aten._softmax, aten.bmm]
        extern_kernels.bmm(buf131, reinterpret_tensor(buf130, (1024, 128, 64), (8192, 64, 1), 0), out=buf132)
        del buf131
        buf133 = empty_strided_cuda((256, 128, 4, 64), (32768, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf132, buf133, 8388608, grid=grid(8388608), stream=stream0)
        buf134 = reinterpret_tensor(buf132, (32768, 256), (256, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (32768, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 256), (1, 256), 0), out=buf134)
        del arg89_1
        buf138 = reinterpret_tensor(buf133, (256, 128, 256), (32768, 256, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf121, buf134, arg90_1, arg91_1, arg92_1, buf138, 32768, 256, grid=grid(32768), stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        buf139 = reinterpret_tensor(buf116, (32768, 2048), (2048, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (32768, 256), (256, 1), 0), reinterpret_tensor(arg93_1, (256, 2048), (1, 256), 0), out=buf139)
        del arg93_1
        buf140 = reinterpret_tensor(buf139, (256, 128, 2048), (262144, 2048, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50], Original ATen: [aten.relu]
        triton_poi_fused_relu_7.run(buf140, arg94_1, 67108864, grid=grid(67108864), stream=stream0)
        del arg94_1
        buf141 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf140, (32768, 2048), (2048, 1), 0), reinterpret_tensor(arg95_1, (2048, 256), (1, 2048), 0), out=buf141)
        del arg95_1
        del buf140
        buf145 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54, hidden_states_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_6.run(buf138, buf141, arg96_1, arg97_1, arg98_1, buf145, 32768, 256, grid=grid(32768), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del buf138
        del buf141
        buf146 = empty_strided_cuda((32768, 10000), (10016, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (32768, 256), (256, 1), 0), reinterpret_tensor(arg1_1, (256, 10000), (1, 256), 0), out=buf146)
        del arg1_1
        del buf145
        buf147 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
        buf148 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_8.run(buf146, buf147, buf148, 32768, 10000, grid=grid(32768), stream=stream0)
        buf149 = empty_strided_cuda((4, ), (1, ), torch.float32)
        buf151 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_9.run(arg99_1, buf146, buf147, buf148, buf149, buf151, 4, 8192, grid=grid(4), stream=stream0)
        del arg99_1
        del buf147
        del buf148
        buf150 = empty_strided_cuda((), (), torch.float32)
        buf153 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_10.run(buf153, buf149, buf151, 1, 4, grid=grid(1), stream=stream0)
        del buf149
        del buf151
    return (buf153, reinterpret_tensor(buf146, (256, 128, 10000), (1282048, 10016, 1), 0), buf4, buf10, buf28, buf34, buf52, buf58, buf76, buf82, buf100, buf106, buf124, buf130, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((10000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1026, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2048, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('Speech2Text2ForCausalLM', benchmark_compiled_module)
