# AOT ID: ['2_inference']
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


# kernel path: /tmp/torchinductor_sahanp/mc/cmcdurdtedemjtfocj77mvzv3htzwftf6sarqsgsrz5hzy5d4ubn.py
# Topologically Sorted Source Nodes: [query_vectors, key_vectors, value_vectors], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   key_vectors => clone_1
#   query_vectors => clone
#   value_vectors => clone_2
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 768
    x1 = (xindex // 768) % 4
    x2 = (xindex // 3072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*x2) + (786432*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp0, None)
    tl.store(out_ptr2 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q7/cq7b7ceb3hy46a7yykuba4kv3w5biaovlbg5wxzd66phx3rexv5t.py
# Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   diagonal_chunked_attention_scores => view_14
# Graph fragment:
#   %view_14 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_13, [48, 2, 512, 64]), kwargs = {})
triton_poi_fused_view_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 48
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (64*(x1 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.125
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2m/c2mxgiibvn7zsaowzaqs6w4baf5uu6bokcrzgx6nz4mymtgf24xe.py
# Topologically Sorted Source Nodes: [hidden_states_2], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   hidden_states_2 => view_11
# Graph fragment:
#   %view_11 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_9, [48, 2, 512, 64]), kwargs = {})
triton_poi_fused_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 48
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (64*(x1 % 12))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xi/cxits5izrvbcybu3nn65cmzgzewx5eolvw4pvwh2rgggqern3vfm.py
# Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   diagonal_chunked_attention_scores => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 3
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x3) + (3072*x1) + (786432*x2)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ib/cib35k7ewisttg6ey3fgrziyk2rw5cinrocn7qb24rh4uslzx2tk.py
# Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   diagonal_chunked_attention_scores => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64) % 3
    y2 = (yindex // 192)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*y2) + (3072*x3) + (786432*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (512*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7c/c7cthirjbdcfjujruvmzspvidjvih4hcqi6wc64r56t3beuavd2d.py
# Topologically Sorted Source Nodes: [diagonal_attention_scores, setitem, setitem_1], Original ATen: [aten.new_zeros, aten.copy]
# Source node to ATen node mapping:
#   diagonal_attention_scores => full
#   setitem => copy
#   setitem_1 => copy_1
# Graph fragment:
#   %full : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([48, 4, 256, 513], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %slice_4), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor, %copy, 3, 256, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %slice_scatter_default, 1, 0, -1), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_24, %slice_18), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int, %copy_1, 2, 256, 9223372036854775807), kwargs = {})
#   %select_scatter_default : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default_1, %slice_scatter_default_2, 1, -1), kwargs = {})
triton_poi_fused_copy_new_zeros_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 98496
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 24624)
    x1 = xindex % 513
    y0 = yindex
    x2 = (xindex // 513) % 48
    x5 = xindex % 24624
    tmp0 = x3
    tmp1 = tl.full([1, 1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tl.full([1, 1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + x1 + (513*y0)) // 512) % 513
    tmp7 = tl.full([1, 1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((512*(((656384 + x1 + (513*y0)) // 512) % 513)) + (262144*((656384 + x1 + (513*y0)) // 262656)) + (786432*x2) + (786432*((656384 + x1 + (513*y0)) // 787968)) + ((x1 + (513*y0)) % 512)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tl.full([1, 1], 3, tl.int64)
    tmp14 = tmp13 < tmp13
    tmp15 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp4
    tmp17 = tmp16 & tmp14
    tmp18 = ((787712 + x1 + (513*y0)) // 512) % 513
    tmp19 = tmp18 < tmp7
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr0 + ((262144*(((787712 + x1 + (513*y0)) // 262656) % 3)) + (786432*(((787712 + x1 + (513*y0) + (787968*x2)) // 787968) % 48)) + ((787712 + x1 + (513*y0)) % 262656)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp17, tmp21, tmp22)
    tmp24 = 0.0
    tmp25 = tl.where(tmp16, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp14, tmp25, tmp26)
    tmp28 = tl.where(tmp14, tmp27, tmp24)
    tmp29 = tl.where(tmp5, tmp12, tmp28)
    tmp30 = tmp0 < tmp13
    tmp31 = tmp16 & tmp30
    tmp32 = (((-256) + x1 + (513*y0) + (262656*x3) + (787968*x2)) // 512) % 513
    tmp33 = tmp32 < tmp7
    tmp34 = tmp33 & tmp31
    tmp35 = tl.load(in_ptr0 + ((262144*((((-256) + x1 + (513*y0) + (262656*x3) + (787968*x2)) // 262656) % 144)) + (((-256) + x1 + (513*y0) + (262656*x3) + (787968*x2)) % 262656)), tmp34 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp31, tmp35, tmp36)
    tmp38 = tl.where(tmp16, tmp37, tmp24)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp30, tmp38, tmp39)
    tmp41 = tl.where(tmp30, tmp40, tmp24)
    tmp42 = tl.where(tmp2, tmp29, tmp41)
    tl.store(out_ptr0 + (x5 + (24640*x3) + (98560*y0)), tmp42, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yn/cynwpp6i346p7a5whzqfk3vfu3hik2r3e7tepc2evwipnlmixa4w.py
# Topologically Sorted Source Nodes: [setitem_2, setitem_3], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_2 => copy_2
#   setitem_3 => copy_3
# Graph fragment:
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_41, %slice_33), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_1, %copy_2, 3, 0, 256), kwargs = {})
#   %slice_scatter_default_4 : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_scatter_default, %slice_scatter_default_3, 1, 1, 9223372036854775807), kwargs = {})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_57, %slice_51), kwargs = {})
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_2, %copy_3, 2, 1, 256), kwargs = {})
#   %slice_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_1, %slice_scatter_default_5, 1, 1, 256), kwargs = {})
#   %select_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default_4, %slice_scatter_default_6, 1, 0), kwargs = {})
triton_poi_fused_copy_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex // 6303744)
    x1 = (xindex // 513) % 256
    x0 = xindex % 513
    x2 = (xindex // 131328) % 48
    x4 = xindex % 131328
    tmp47 = tl.load(in_ptr1 + (x0 + (513*x2) + (98560*x1)), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr1 + (x0 + (513*x2) + (24640*x3) + (98560*x1)), None)
    tmp0 = x3
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp4
    tmp8 = tl.full([1], 256, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = (((-256) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (787968*x2)) // 262656) % 144)) + (((-256) + x0 + (513*x1) + (787968*x2)) % 262656)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp11, tmp16, tmp17)
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp19 >= tmp4
    tmp21 = tmp20 & tmp5
    tmp22 = tmp9 & tmp21
    tmp23 = (((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp24 = tmp23 < tmp13
    tmp25 = tmp24 & tmp22
    tmp26 = tl.load(in_ptr0 + ((512*((((-131584) + x4 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x4 + (787968*x2)) // 262656) % 144)) + (x4 % 512)), tmp25, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp22, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (x0 + (513*x2) + (98560*x1)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp9, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp21, tmp30, tmp31)
    tmp33 = tl.load(in_ptr1 + (x0 + (513*x2) + (98560*x1)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp20, tmp32, tmp33)
    tmp35 = tl.where(tmp10, tmp18, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tmp9 & tmp20
    tmp39 = tmp24 & tmp38
    tmp40 = tl.load(in_ptr0 + ((512*((((-131584) + x4 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x4 + (787968*x2)) // 262656) % 144)) + (x4 % 512)), tmp39, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp38, tmp40, tmp41)
    tmp43 = tl.load(in_ptr1 + (x0 + (513*x2) + (98560*x1)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.where(tmp9, tmp42, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp20, tmp44, tmp45)
    tmp48 = tl.where(tmp20, tmp46, tmp47)
    tmp49 = tl.where(tmp5, tmp37, tmp48)
    tmp50 = tmp0 >= tmp4
    tmp51 = tmp9 & tmp50
    tmp52 = (((-131584) + x0 + (513*x1) + (262656*x3) + (787968*x2)) // 512) % 513
    tmp53 = tmp52 < tmp13
    tmp54 = tmp53 & tmp51
    tmp55 = tl.load(in_ptr0 + ((512*((((-131584) + x4 + (262656*x3) + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x4 + (262656*x3) + (787968*x2)) // 262656) % 144)) + (x4 % 512)), tmp54, other=0.0)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp51, tmp55, tmp56)
    tmp58 = tl.load(in_ptr1 + (x0 + (513*x2) + (24640*x3) + (98560*x1)), tmp50, other=0.0)
    tmp59 = tl.where(tmp9, tmp57, tmp58)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp50, tmp59, tmp60)
    tmp63 = tl.where(tmp50, tmp61, tmp62)
    tmp64 = tl.where(tmp2, tmp49, tmp63)
    tl.store(out_ptr0 + (x4 + (131328*x3) + (525312*x2)), tmp64, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gm/cgmxfs7wpdrdadaicrqelkm457s7e6z57ptpb4qeo5qczkbw3mag.py
# Topologically Sorted Source Nodes: [bool_2, full_like_1, where_1, setitem_5], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
# Source node to ATen node mapping:
#   bool_2 => convert_element_type_1
#   full_like_1 => full_default_3
#   setitem_5 => copy_5
#   where_1 => where_2
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_1, torch.bool), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 12, 257], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_1, %full_default_3, %slice_95), kwargs = {})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_103, %where_2), kwargs = {})
#   %slice_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_4, %copy_5, 3, -257, 9223372036854775807), kwargs = {})
triton_poi_fused__to_copy_copy_full_like_where_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_full_like_where_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    tmp43 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), None)
    tmp0 = x0
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 512 + ((-1)*x0) + ((-1)*x1)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 <= tmp4
    tmp6 = 1.0
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = (tmp8 != 0)
    tmp10 = 768 + x1
    tmp11 = tmp10 < tmp1
    tmp12 = tmp11 & tmp2
    tmp13 = tl.full([1], 257, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp14 & tmp12
    tmp16 = 513 + x0 + x1
    tmp17 = tmp16 <= tmp4
    tmp18 = tl.where(tmp17, tmp6, tmp7)
    tmp19 = (tmp18 != 0)
    tmp20 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), tmp15, other=0.0)
    tmp21 = float("-inf")
    tmp22 = tl.where(tmp19, tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), tmp12, other=0.0)
    tmp26 = tl.where(tmp14, tmp24, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp12, tmp26, tmp27)
    tmp29 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), tmp2, other=0.0)
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp31 = tl.where(tmp9, tmp21, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp2, tmp31, tmp32)
    tmp34 = tmp14 & tmp11
    tmp35 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), tmp34, other=0.0)
    tmp36 = tl.where(tmp19, tmp21, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp34, tmp36, tmp37)
    tmp39 = tl.load(in_ptr0 + (393984 + x3 + (525312*x2)), tmp11, other=0.0)
    tmp40 = tl.where(tmp14, tmp38, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp11, tmp40, tmp41)
    tmp44 = tl.where(tmp11, tmp42, tmp43)
    tmp45 = tl.where(tmp2, tmp33, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dl/cdlstz72dda6b36trb5vx4zoxuu7uohgyfhil6tizb6bvkfvnlee.py
# Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   hidden_states_3 => view_34
# Graph fragment:
#   %view_34 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_32, [4, 2, 512, 1]), kwargs = {})
triton_poi_fused_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtco2b5xohw3ge65rhx37urc33zjuedtgi3or3u7rkx4wqz3dfo.py
# Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_17], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   hidden_states_17 => view_107
#   hidden_states_4 => view_35
# Graph fragment:
#   %view_35 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_33, [4, 2, 512, 1]), kwargs = {})
#   %view_107 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_105, [4, 2, 512, 1]), kwargs = {})
triton_poi_fused_view_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp2, tmp4, tmp3)
    tl.store(out_ptr0 + (x0), tmp5, None)
    tl.store(out_ptr1 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yk/cykphbgl6qmmnhzpuiisiikzgsg75mtaxnykoah2kep24f7ayqa7.py
# Topologically Sorted Source Nodes: [diagonal_attention_scores_2, setitem_6, setitem_7], Original ATen: [aten.new_zeros, aten.copy]
# Source node to ATen node mapping:
#   diagonal_attention_scores_2 => full_5
#   setitem_6 => copy_6
#   setitem_7 => copy_7
# Graph fragment:
#   %full_5 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 256, 513], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_120, %slice_116), kwargs = {})
#   %slice_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_5, %copy_6, 3, 256, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_12 : [num_users=4] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_5, %slice_scatter_default_11, 1, 0, -1), kwargs = {})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_136, %slice_130), kwargs = {})
#   %slice_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_2, %copy_7, 2, 256, 9223372036854775807), kwargs = {})
#   %select_scatter_default_2 : [num_users=4] = call_function[target=torch.ops.aten.select_scatter.default](args = (%slice_scatter_default_12, %slice_scatter_default_13, 1, -1), kwargs = {})
triton_poi_fused_copy_new_zeros_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2101248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex // 525312)
    x0 = xindex % 513
    x2 = (xindex // 2052) % 256
    x1 = (xindex // 513) % 4
    tmp0 = x3
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + x0 + (513*x2)) // 512) % 513
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((256*((656384 + x0 + (513*x2)) // 262656)) + (1024*x1) + (1024*((656384 + x0 + (513*x2)) // 787968)) + (((656384 + x0 + (513*x2)) // 512) % 513)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + ((256*((656384 + x0 + (513*x2)) // 262656)) + (1024*x1) + (1024*((656384 + x0 + (513*x2)) // 787968)) + ((x0 + (513*x2)) % 512)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.full([1], 3, tl.int64)
    tmp18 = tmp17 < tmp17
    tmp19 = tmp5 & tmp18
    tmp20 = ((787712 + x0 + (513*x2)) // 512) % 513
    tmp21 = tmp20 < tmp7
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + ((256*(((787712 + x0 + (513*x2)) // 262656) % 3)) + (1024*(((787712 + x0 + (513*x2) + (787968*x1)) // 787968) % 4)) + (((787712 + x0 + (513*x2)) // 512) % 513)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + ((256*(((787712 + x0 + (513*x2)) // 262656) % 3)) + (1024*(((787712 + x0 + (513*x2) + (787968*x1)) // 787968) % 4)) + ((787712 + x0 + (513*x2)) % 512)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = 0.0
    tmp31 = tl.where(tmp5, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp18, tmp31, tmp32)
    tmp34 = tl.where(tmp18, tmp33, tmp30)
    tmp35 = tl.where(tmp5, tmp16, tmp34)
    tmp36 = tmp0 < tmp17
    tmp37 = tmp5 & tmp36
    tmp38 = (((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 512) % 513
    tmp39 = tmp38 < tmp7
    tmp40 = tmp39 & tmp37
    tmp41 = tl.load(in_ptr0 + ((256*((((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 787968) % 4)) + ((((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 512) % 513)), tmp40, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr1 + ((256*((((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 262656) % 3)) + (1024*((((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) // 787968) % 4)) + (((-256) + x0 + (513*x2) + (262656*x3) + (787968*x1)) % 512)), tmp40, other=0.0)
    tmp43 = tmp41 * tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp40, tmp43, tmp44)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tl.where(tmp5, tmp47, tmp30)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp36, tmp48, tmp49)
    tmp51 = tl.where(tmp36, tmp50, tmp30)
    tmp52 = tl.where(tmp2, tmp35, tmp51)
    tl.store(out_ptr0 + (x0 + (513*x2) + (131328*x1) + (525312*x3)), tmp52, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nt/cnt7zc4nzwfl7tppz67kqpjtrrg2zihm3w3tuwm7j6xc5xzyq5wh.py
# Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_9 => copy_9
# Graph fragment:
#   %copy_9 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_169, %slice_163), kwargs = {})
#   %slice_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_7, %copy_9, 2, 1, 256), kwargs = {})
#   %slice_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_3, %slice_scatter_default_16, 1, 1, 256), kwargs = {})
triton_poi_fused_copy_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 256
    x2 = xindex
    y1 = (yindex // 256)
    y3 = yindex
    tmp56 = tl.load(in_ptr2 + (x2 + (513*y3)), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1, 1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x2 + (513*y0) + (787968*y1)) // 512) % 513
    tmp10 = tl.full([1, 1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((256*((((-256) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-256) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((((-256) + x2 + (513*y0) + (787968*y1)) // 512) % 513)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + ((256*((((-256) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-256) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + (((-256) + x2 + (513*y0) + (787968*y1)) % 512)), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp21 & tmp2
    tmp23 = tmp6 & tmp22
    tmp24 = (((-131584) + x2 + (513*y0) + (787968*y1)) // 512) % 513
    tmp25 = tmp24 < tmp10
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr0 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((((-131584) + x2 + (513*y0) + (787968*y1)) // 512) % 513)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr1 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((x2 + (513*y0)) % 512)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp26, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tl.load(in_ptr2 + (x2 + (513*y3)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp6, tmp33, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp22, tmp35, tmp36)
    tmp38 = tl.load(in_ptr2 + (x2 + (513*y3)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp7, tmp19, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp2, tmp40, tmp41)
    tmp43 = tmp6 & tmp21
    tmp44 = tmp25 & tmp43
    tmp45 = tl.load(in_ptr0 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((((-131584) + x2 + (513*y0) + (787968*y1)) // 512) % 513)), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr1 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((x2 + (513*y0)) % 512)), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 * tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp44, tmp47, tmp48)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp43, tmp49, tmp50)
    tmp52 = tl.load(in_ptr2 + (x2 + (513*y3)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.where(tmp6, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp21, tmp53, tmp54)
    tmp57 = tl.where(tmp21, tmp55, tmp56)
    tmp58 = tl.where(tmp2, tmp42, tmp57)
    tl.store(out_ptr0 + (x2 + (513*y3)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cb/ccbw24w3pnfwijqbkpx5txwq2uyixusudbgzwsu7uetblbqlfyhd.py
# Topologically Sorted Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# Source node to ATen node mapping:
#   bool_3 => convert_element_type_3
#   full_like_2 => full_default_7
#   where_2 => where_5
# Graph fragment:
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_2, torch.bool), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 256, 1, 257], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_3, %full_default_7, %slice_184), kwargs = {})
triton_poi_fused__to_copy_full_like_where_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y3 = yindex
    y1 = (yindex // 256)
    tmp9 = tl.load(in_ptr0 + (x2 + (513*y3)), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x2 + (513*y3)), xmask, eviction_policy='evict_last')
    tmp0 = (-255) + x2 + y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x2 + (513*y0) + (787968*y1)) // 512) % 513
    tmp17 = tl.full([1, 1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((((-131584) + x2 + (513*y0) + (787968*y1)) // 512) % 513)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr2 + ((256*((((-131584) + x2 + (513*y0) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*y0) + (787968*y1)) // 787968) % 4)) + ((x2 + (513*y0)) % 512)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.load(in_ptr3 + (x2 + (513*y3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp14, tmp26, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp32 = tl.where(tmp11, tmp30, tmp31)
    tmp33 = tl.where(tmp8, tmp9, tmp32)
    tmp34 = float("-inf")
    tmp35 = tl.where(tmp6, tmp34, tmp33)
    tl.store(out_ptr0 + (x2 + (257*y3)), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qt/cqtrnlnu4itjmk2ce4wttzxj5aqxjayytulmpxmwfemirluleuug.py
# Topologically Sorted Source Nodes: [setitem_10], Original ATen: [aten.copy]
# Source node to ATen node mapping:
#   setitem_10 => copy_10
# Graph fragment:
#   %copy_10 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_192, %where_5), kwargs = {})
#   %slice_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_tensor_8, %copy_10, 3, 0, 257), kwargs = {})
#   %slice_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute_37, %slice_scatter_default_18, 1, 0, 256), kwargs = {})
triton_poi_fused_copy_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1024
    x2 = xindex
    y1 = (yindex // 1024)
    y3 = yindex
    tmp40 = tl.load(in_ptr1 + (x2 + (513*(y0 % 256)) + (131328*y1)), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr4 + (x2 + (513*(y0 % 256)) + (131328*y1) + (525312*(y0 // 256))), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp4 = tl.full([1, 1], 257, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x2 + (257*y0) + (65792*y1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.broadcast_to((y0 // 256), [XBLOCK, YBLOCK])
    tmp9 = tl.full([1, 1], 0, tl.int32)
    tmp10 = tmp8 == tmp9
    tmp11 = tl.load(in_ptr1 + (x2 + (513*(y0 % 256)) + (131328*y1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 1, tl.int64)
    tmp13 = tmp8 >= tmp12
    tmp14 = tmp13 & tmp2
    tmp15 = tmp3 < tmp1
    tmp16 = tmp15 & tmp14
    tmp17 = (((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 512) % 513
    tmp18 = tl.full([1, 1], 512, tl.int64)
    tmp19 = tmp17 < tmp18
    tmp20 = tmp19 & tmp16
    tmp21 = tl.load(in_ptr2 + ((256*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 787968) % 4)) + ((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 512) % 513)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + ((256*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 787968) % 4)) + ((x2 + (513*(y0 % 256))) % 512)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp16, tmp25, tmp26)
    tmp28 = tl.load(in_ptr4 + (x2 + (513*(y0 % 256)) + (131328*y1) + (525312*(y0 // 256))), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp15, tmp27, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp14, tmp29, tmp30)
    tmp32 = tl.load(in_ptr4 + (x2 + (513*(y0 % 256)) + (131328*y1) + (525312*(y0 // 256))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.where(tmp13, tmp31, tmp32)
    tmp34 = tl.where(tmp10, tmp11, tmp33)
    tmp35 = tl.where(tmp5, tmp7, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp38 = (y0 // 256)
    tmp39 = tmp38 == tmp9
    tmp41 = tmp38 >= tmp12
    tmp42 = tmp15 & tmp41
    tmp43 = tmp19 & tmp42
    tmp44 = tl.load(in_ptr2 + ((256*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 787968) % 4)) + ((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 512) % 513)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr3 + ((256*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 262656) % 3)) + (1024*((((-131584) + x2 + (513*(y0 % 256)) + (262656*(y0 // 256)) + (787968*y1)) // 787968) % 4)) + ((x2 + (513*(y0 % 256))) % 512)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp43, tmp46, tmp47)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp42, tmp48, tmp49)
    tmp51 = tl.load(in_ptr4 + (x2 + (513*(y0 % 256)) + (131328*y1) + (525312*(y0 // 256))), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.where(tmp15, tmp50, tmp51)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp41, tmp52, tmp53)
    tmp56 = tl.where(tmp41, tmp54, tmp55)
    tmp57 = tl.where(tmp39, tmp40, tmp56)
    tmp58 = tl.where(tmp2, tmp37, tmp57)
    tl.store(out_ptr0 + (x2 + (513*y3)), tmp58, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zi/czizp2day3mz7hqphabfvgvibqg6f4ejcnqwkksyqgl3zwncyjml.py
# Topologically Sorted Source Nodes: [attn_scores], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   attn_scores => add_5
# Graph fragment:
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_45, %permute_46), kwargs = {})
triton_poi_fused_add_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 513) % 1024
    x4 = xindex % 525312
    x5 = (xindex // 525312)
    x0 = xindex % 513
    x3 = (xindex // 6303744)
    x2 = (xindex // 525312) % 12
    tmp27 = tl.load(in_ptr1 + (x0 + (513*x1) + (131328*((x1 % 256) // 256)) + (525312*x5)), None)
    tmp44 = tl.load(in_ptr2 + (x4 + (525312*x3)), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-393984) + x4 + (131328*x5)), tmp2, other=0.0)
    tmp4 = x1 + (256*((x1 % 256) // 256))
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = x0
    tmp8 = tl.full([1], 257, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (-255) + x0 + x1 + (256*((x1 % 256) // 256))
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp11 <= tmp12
    tmp14 = 1.0
    tmp15 = 0.0
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = (tmp16 != 0)
    tmp18 = tl.load(in_ptr1 + (x0 + (513*x1) + (131328*((x1 % 256) // 256)) + (525312*x5)), tmp10, other=0.0)
    tmp19 = float("-inf")
    tmp20 = tl.where(tmp17, tmp19, tmp18)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp10, tmp20, tmp21)
    tmp23 = tl.load(in_ptr1 + (x0 + (513*x1) + (131328*((x1 % 256) // 256)) + (525312*x5)), tmp6, other=0.0)
    tmp24 = tl.where(tmp9, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp2, tmp3, tmp28)
    tmp30 = tmp7 >= tmp5
    tmp31 = tmp30 & tmp2
    tmp32 = 1280 + ((-1)*x0) + ((-1)*x1)
    tmp33 = tmp32 <= tmp12
    tmp34 = tl.where(tmp33, tmp14, tmp15)
    tmp35 = (tmp34 != 0)
    tmp36 = tl.load(in_ptr2 + (x4 + (525312*x3)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.where(tmp35, tmp19, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp31, tmp37, tmp38)
    tmp40 = tl.load(in_ptr2 + (x4 + (525312*x3)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.where(tmp30, tmp39, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp2, tmp41, tmp42)
    tmp45 = tl.where(tmp2, tmp43, tmp44)
    tmp46 = tmp29 + tmp45
    tl.store(out_ptr0 + (x0 + (513*x2) + (6176*x1) + (6324224*x3)), tmp46, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dj/cdjddikzmas2ha3t2eeiytj4peoxlwskvapquung2h767kmkmgk2.py
# Topologically Sorted Source Nodes: [attn_probs], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_probs => amax, clone_5, exp, sub_4, sum_1
# Graph fragment:
#   %clone_5 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_48,), kwargs = {memory_format: torch.contiguous_format})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_5, [-1], True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_red_fused__softmax_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 12
    x1 = (xindex // 12)
    _tmp2 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (513*x0) + (6176*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = triton_helpers.maximum(_tmp2, tmp1)
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = triton_helpers.max2(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp4 = tl.load(in_ptr0 + (r2 + (513*x0) + (6176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3r/c3rvakgfxdid4yzo5vekjnbmka6wlpfxxwq5spagx366mwcjmrla.py
# Topologically Sorted Source Nodes: [padded_value], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   padded_value => constant_pad_nd_2
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_57, [0, 0, 256, 256], -1.0), kwargs = {})
triton_poi_fused_constant_pad_nd_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64) % 1536
    x0 = xindex % 64
    x2 = (xindex // 98304)
    x3 = xindex
    tmp0 = (-256) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-786432) + x0 + (64*x2) + (3072*x1)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full(tmp8.shape, -1.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpad6s3yeo4n2ibblvktm2svsqurln5a7yb57vzyyykbhvxjkea.py
# Topologically Sorted Source Nodes: [chunked_hidden_states], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   chunked_hidden_states => constant_pad_nd_3
# Graph fragment:
#   %constant_pad_nd_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_56, [0, 257], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37847040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 770
    x1 = (xindex // 770) % 48
    x2 = (xindex // 36960)
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (1024*(x1 // 12))), tmp2, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.load(in_ptr1 + (x0 + (513*(x1 % 12)) + (6176*x2) + (6324224*(x1 // 12)) + (6324224*((x1 % 12) // 12))), tmp2, other=0.0)
    tmp5 = tl.load(in_ptr2 + ((12*x2) + (12288*(x1 // 12)) + (x1 % 12)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp4 - tmp5
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tl.load(in_ptr3 + ((12*x2) + (12288*(x1 // 12)) + (x1 % 12)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp3, tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tl.store(out_ptr0 + (x0 + (770*x2) + (788480*x1)), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/py/cpy5zn2vgma4atrachvrekkroboy56ht6tifzvqvekgme3m2bua3.py
# Topologically Sorted Source Nodes: [context], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   context => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_55,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9437184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 49152
    x1 = (xindex // 49152) % 4
    x2 = (xindex // 196608)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16384*x1) + (98304*x2)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sh/cshxkm45mervtyvy6jei5c4tdcwsvlz6wwlahu5tloesavayaz54.py
# Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   hidden_states_5 => clone_10
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_59,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (786432*x2) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mu/cmuq3sracoofpzyy6qgpwgyezvpsyea2mrw6cf3ibigjvn5lzbst.py
# Topologically Sorted Source Nodes: [hidden_states_5, add_3, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_3 => add_8
#   hidden_states_5 => add_7
#   hidden_states_7 => add_10, add_9, mul_1, mul_2, rsqrt, sub_6, var_mean
# Graph fragment:
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_67, %arg10_1), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %arg0_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %arg11_1), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg12_1), kwargs = {})
triton_per_fused_add_native_layer_norm_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_20', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ll/clle65qffpzvch4ywhfxstlk2lyl5yappru5gltqoodwhgerxyup.py
# Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_9 => add_11, erf, mul_3, mul_4, mul_5
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_69, 0.5), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_69, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_4,), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %add_11), kwargs = {})
triton_poi_fused_gelu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024, 768), (786432, 768, 1))
    assert_size_stride(arg1_1, (768, 768), (768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, 768), (768, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, 768), (768, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (4, 1024), (1024, 1))
    assert_size_stride(arg8_1, (4, 1024), (1024, 1))
    assert_size_stride(arg9_1, (768, 768), (768, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (3072, 768), (768, 1))
    assert_size_stride(arg14_1, (3072, ), (1, ))
    assert_size_stride(arg15_1, (768, 3072), (3072, 1))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, 768), (768, 1))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, 768), (768, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, 768), (768, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, 768), (768, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (3072, 768), (768, 1))
    assert_size_stride(arg30_1, (3072, ), (1, ))
    assert_size_stride(arg31_1, (768, 3072), (3072, 1))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, 768), (768, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, 768), (768, 1))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, 768), (768, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (3072, 768), (768, 1))
    assert_size_stride(arg46_1, (3072, ), (1, ))
    assert_size_stride(arg47_1, (768, 3072), (3072, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (768, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, 768), (768, 1))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, 768), (768, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, 768), (768, 1))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, 768), (768, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (3072, 768), (768, 1))
    assert_size_stride(arg62_1, (3072, ), (1, ))
    assert_size_stride(arg63_1, (768, 3072), (3072, 1))
    assert_size_stride(arg64_1, (768, ), (1, ))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, 768), (768, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, 768), (768, 1))
    assert_size_stride(arg72_1, (768, ), (1, ))
    assert_size_stride(arg73_1, (768, 768), (768, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, ), (1, ))
    assert_size_stride(arg77_1, (3072, 768), (768, 1))
    assert_size_stride(arg78_1, (3072, ), (1, ))
    assert_size_stride(arg79_1, (768, 3072), (3072, 1))
    assert_size_stride(arg80_1, (768, ), (1, ))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, 768), (768, 1))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, 768), (768, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
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
    assert_size_stride(arg99_1, (768, 768), (768, 1))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, 768), (768, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, 768), (768, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 768), (768, 1))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (3072, 768), (768, 1))
    assert_size_stride(arg110_1, (3072, ), (1, ))
    assert_size_stride(arg111_1, (768, 3072), (3072, 1))
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
    assert_size_stride(arg141_1, (3072, 768), (768, 1))
    assert_size_stride(arg142_1, (3072, ), (1, ))
    assert_size_stride(arg143_1, (768, 3072), (3072, 1))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, 768), (768, 1))
    assert_size_stride(arg150_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, 768), (768, 1))
    assert_size_stride(arg152_1, (768, ), (1, ))
    assert_size_stride(arg153_1, (768, 768), (768, 1))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (3072, 768), (768, 1))
    assert_size_stride(arg158_1, (3072, ), (1, ))
    assert_size_stride(arg159_1, (768, 3072), (3072, 1))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (768, 768), (768, 1))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, 768), (768, 1))
    assert_size_stride(arg168_1, (768, ), (1, ))
    assert_size_stride(arg169_1, (768, 768), (768, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (3072, 768), (768, 1))
    assert_size_stride(arg174_1, (3072, ), (1, ))
    assert_size_stride(arg175_1, (768, 3072), (3072, 1))
    assert_size_stride(arg176_1, (768, ), (1, ))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (768, 768), (768, 1))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, 768), (768, 1))
    assert_size_stride(arg182_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, 768), (768, 1))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (768, 768), (768, 1))
    assert_size_stride(arg186_1, (768, ), (1, ))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, ), (1, ))
    assert_size_stride(arg189_1, (3072, 768), (768, 1))
    assert_size_stride(arg190_1, (3072, ), (1, ))
    assert_size_stride(arg191_1, (768, 3072), (3072, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 4, 768), (3072, 768, 1), torch.float32)
        buf3 = empty_strided_cuda((1024, 4, 768), (3072, 768, 1), torch.float32)
        buf21 = empty_strided_cuda((1024, 4, 768), (3072, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_vectors, key_vectors, value_vectors], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(arg0_1, buf0, buf3, buf21, 3145728, grid=grid(3145728), stream=stream0)
        buf1 = empty_strided_cuda((4096, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_vectors], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (4096, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 768), (1, 768), 0), out=buf1)
        del arg1_1
        buf2 = reinterpret_tensor(buf1, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf2, arg2_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg2_1
        buf4 = reinterpret_tensor(buf0, (4096, 768), (768, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [key_vectors], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), reinterpret_tensor(arg3_1, (768, 768), (1, 768), 0), out=buf4)
        del arg3_1
        buf5 = reinterpret_tensor(buf4, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_2], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf5, arg4_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg4_1
        buf6 = empty_strided_cuda((48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf2, buf6, 4718592, grid=grid(4718592), stream=stream0)
        buf7 = empty_strided_cuda((48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf5, buf7, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf8 = empty_strided_cuda((144, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf7, (144, 64, 512), (32768, 512, 1), 0), out=buf8)
        buf9 = empty_strided_cuda((48, 4, 256, 513), (513, 24640, 98560, 1), torch.float32)
        # Topologically Sorted Source Nodes: [diagonal_attention_scores, setitem, setitem_1], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf8, buf9, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf10 = empty_strided_cuda((48, 4, 256, 513), (525312, 131328, 513, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_2, setitem_3], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf8, buf9, buf10, 25214976, grid=grid(25214976), stream=stream0)
        buf11 = empty_strided_cuda((4, 256, 12, 513), (1575936, 513, 131328, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bool_2, full_like_1, where_1, setitem_5], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf10, buf11, 6303744, grid=grid(6303744), stream=stream0)
        buf12 = empty_strided_cuda((4, 2, 512, 1), (1024, 512, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf12, 4096, grid=grid(4096), stream=stream0)
        buf13 = empty_strided_cuda((4, 2, 512, 1), (1024, 512, 1, 1), torch.float32)
        buf53 = empty_strided_cuda((4, 2, 512, 1), (1024, 512, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_17], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf13, buf53, 4096, grid=grid(4096), stream=stream0)
        buf14 = empty_strided_cuda((4, 4, 256, 513), (131328, 525312, 513, 1), torch.float32)
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_2, setitem_6, setitem_7], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf12, buf13, buf14, 2101248, grid=grid(2101248), stream=stream0)
        buf15 = empty_strided_cuda((4, 256, 513), (131328, 513, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_9], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf12, buf13, buf14, buf15, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf16 = empty_strided_cuda((4, 256, 1, 257), (65792, 257, 263168, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bool_3, full_like_2, where_2], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf15, buf12, buf13, buf14, buf16, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf17 = empty_strided_cuda((4, 1024, 1, 513), (525312, 513, 2101248, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_10], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf16, buf15, buf12, buf13, buf14, buf17, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf18 = empty_strided_cuda((4, 1024, 12, 513), (6324224, 6176, 513, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_scores], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf11, buf10, buf17, buf18, 25214976, grid=grid(25214976), stream=stream0)
        buf19 = empty_strided_cuda((4, 1024, 12, 1), (12288, 12, 1, 49152), torch.float32)
        buf20 = empty_strided_cuda((4, 1024, 12, 1), (12288, 12, 1, 49152), torch.float32)
        # Topologically Sorted Source Nodes: [attn_probs], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf18, buf19, buf20, 49152, 513, grid=grid(49152), stream=stream0)
        buf22 = reinterpret_tensor(buf5, (4096, 768), (768, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [value_vectors], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (4096, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf22)
        del arg5_1
        buf23 = reinterpret_tensor(buf7, (48, 1536, 64), (98304, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [padded_value], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf22, arg6_1, buf23, 4718592, grid=grid(4718592), stream=stream0)
        del arg6_1
        buf24 = empty_strided_cuda((48, 4, 256, 770), (788480, 197120, 770, 1), torch.float32)
        # Topologically Sorted Source Nodes: [chunked_hidden_states], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf18, buf19, buf20, buf24, 37847040, grid=grid(37847040), stream=stream0)
        buf25 = empty_strided_cuda((48, 4, 768, 64, 1), (196608, 49152, 64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf23, buf25, 9437184, grid=grid(9437184), stream=stream0)
        buf26 = reinterpret_tensor(buf22, (192, 256, 64), (16384, 64, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf25, (192, 768, 64), (49152, 64, 1), 0), out=buf26)
        buf27 = reinterpret_tensor(buf21, (4, 1024, 768), (786432, 768, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf26, buf27, 3145728, grid=grid(3145728), stream=stream0)
        buf28 = reinterpret_tensor(buf26, (4096, 768), (768, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), out=buf28)
        del arg9_1
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5, add_3, hidden_states_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf28, arg10_1, arg0_1, arg11_1, arg12_1, buf32, 4096, 768, grid=grid(4096), stream=stream0)
        del arg0_1
        del arg10_1
        del arg11_1
        del arg12_1
        buf33 = empty_strided_cuda((4096, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf32, (4096, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 3072), (1, 768), 0), out=buf33)
        del arg13_1
        buf34 = reinterpret_tensor(buf33, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf34, arg14_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg14_1
        buf35 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf34, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg15_1, (3072, 768), (1, 3072), 0), out=buf35)
        del arg15_1
        buf39 = reinterpret_tensor(buf2, (4, 1024, 768), (786432, 768, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [add_4, hidden_states_12], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf35, arg16_1, buf32, arg17_1, arg18_1, buf39, 4096, 768, grid=grid(4096), stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        buf40 = reinterpret_tensor(buf35, (1024, 4, 768), (3072, 768, 1), 0); del buf35  # reuse
        buf43 = reinterpret_tensor(buf32, (1024, 4, 768), (3072, 768, 1), 0); del buf32  # reuse
        buf61 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_3, key_vectors_2, value_vectors_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf39, buf40, buf43, buf61, 3145728, grid=grid(3145728), stream=stream0)
        buf41 = empty_strided_cuda((4096, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_vectors_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (4096, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 768), (1, 768), 0), out=buf41)
        del arg19_1
        buf42 = reinterpret_tensor(buf41, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_2], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf42, arg20_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg20_1
        buf44 = reinterpret_tensor(buf40, (4096, 768), (768, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (4096, 768), (768, 1), 0), reinterpret_tensor(arg21_1, (768, 768), (1, 768), 0), out=buf44)
        del arg21_1
        buf45 = reinterpret_tensor(buf44, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf45, arg22_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg22_1
        buf46 = reinterpret_tensor(buf23, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf42, buf46, 4718592, grid=grid(4718592), stream=stream0)
        buf47 = reinterpret_tensor(buf6, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf45, buf47, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf48 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf47, (144, 64, 512), (32768, 512, 1), 0), out=buf48)
        buf49 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_4, setitem_12, setitem_13], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf48, buf49, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf50 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [setitem_14, setitem_15], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf48, buf49, buf50, 25214976, grid=grid(25214976), stream=stream0)
        buf51 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [bool_6, full_like_5, where_5, setitem_17], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf50, buf51, 6303744, grid=grid(6303744), stream=stream0)
        buf52 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf52, 4096, grid=grid(4096), stream=stream0)
        buf54 = reinterpret_tensor(buf17, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_6, setitem_18, setitem_19], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf52, buf53, buf54, 2101248, grid=grid(2101248), stream=stream0)
        buf55 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [setitem_21], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf52, buf53, buf54, buf55, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf56 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [bool_7, full_like_6, where_6], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf55, buf52, buf53, buf54, buf56, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf57 = reinterpret_tensor(buf14, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [setitem_22], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf56, buf55, buf52, buf53, buf54, buf57, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf58 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_1], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf51, buf50, buf57, buf58, 25214976, grid=grid(25214976), stream=stream0)
        buf59 = buf20; del buf20  # reuse
        buf60 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_4], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf58, buf59, buf60, 49152, 513, grid=grid(49152), stream=stream0)
        buf62 = reinterpret_tensor(buf45, (4096, 768), (768, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (4096, 768), (768, 1), 0), reinterpret_tensor(arg23_1, (768, 768), (1, 768), 0), out=buf62)
        del arg23_1
        buf63 = reinterpret_tensor(buf47, (48, 1536, 64), (98304, 64, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [padded_value_1], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf62, arg24_1, buf63, 4718592, grid=grid(4718592), stream=stream0)
        del arg24_1
        buf64 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf58, buf59, buf60, buf64, 37847040, grid=grid(37847040), stream=stream0)
        buf65 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [context_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf63, buf65, 9437184, grid=grid(9437184), stream=stream0)
        buf66 = reinterpret_tensor(buf62, (192, 256, 64), (16384, 64, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [context_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf65, (192, 768, 64), (49152, 64, 1), 0), out=buf66)
        buf67 = reinterpret_tensor(buf61, (4, 1024, 768), (786432, 768, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf66, buf67, 3145728, grid=grid(3145728), stream=stream0)
        buf68 = reinterpret_tensor(buf66, (4096, 768), (768, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (4096, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), out=buf68)
        del arg25_1
        buf72 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, add_8, hidden_states_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf68, arg26_1, buf39, arg27_1, arg28_1, buf72, 4096, 768, grid=grid(4096), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        buf73 = reinterpret_tensor(buf34, (4096, 3072), (3072, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf72, (4096, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 3072), (1, 768), 0), out=buf73)
        del arg29_1
        buf74 = reinterpret_tensor(buf73, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_22], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf74, arg30_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg30_1
        buf75 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg31_1, (3072, 768), (1, 3072), 0), out=buf75)
        del arg31_1
        buf79 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [add_9, hidden_states_25], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf75, arg32_1, buf72, arg33_1, arg34_1, buf79, 4096, 768, grid=grid(4096), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        buf80 = reinterpret_tensor(buf75, (1024, 4, 768), (3072, 768, 1), 0); del buf75  # reuse
        buf83 = reinterpret_tensor(buf72, (1024, 4, 768), (3072, 768, 1), 0); del buf72  # reuse
        buf101 = reinterpret_tensor(buf42, (1024, 4, 768), (3072, 768, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_6, key_vectors_4, value_vectors_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf79, buf80, buf83, buf101, 3145728, grid=grid(3145728), stream=stream0)
        buf81 = reinterpret_tensor(buf43, (4096, 768), (768, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (4096, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 768), (1, 768), 0), out=buf81)
        del arg35_1
        buf82 = reinterpret_tensor(buf81, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf82, arg36_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg36_1
        buf84 = reinterpret_tensor(buf80, (4096, 768), (768, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (4096, 768), (768, 1), 0), reinterpret_tensor(arg37_1, (768, 768), (1, 768), 0), out=buf84)
        del arg37_1
        buf85 = reinterpret_tensor(buf84, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf85, arg38_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg38_1
        buf86 = reinterpret_tensor(buf63, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf82, buf86, 4718592, grid=grid(4718592), stream=stream0)
        buf87 = reinterpret_tensor(buf46, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf85, buf87, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf88 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf87, (144, 64, 512), (32768, 512, 1), 0), out=buf88)
        buf89 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_8, setitem_24, setitem_25], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf88, buf89, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf90 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [setitem_26, setitem_27], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf88, buf89, buf90, 25214976, grid=grid(25214976), stream=stream0)
        buf91 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [bool_10, full_like_9, where_9, setitem_29], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf90, buf91, 6303744, grid=grid(6303744), stream=stream0)
        buf92 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf92, 4096, grid=grid(4096), stream=stream0)
        buf93 = buf52; del buf52  # reuse
        buf133 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_43], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf93, buf133, 4096, grid=grid(4096), stream=stream0)
        buf94 = reinterpret_tensor(buf57, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_10, setitem_30, setitem_31], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf92, buf93, buf94, 2101248, grid=grid(2101248), stream=stream0)
        buf95 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [setitem_33], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf92, buf93, buf94, buf95, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf96 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [bool_11, full_like_10, where_10], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf95, buf92, buf93, buf94, buf96, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf97 = reinterpret_tensor(buf54, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [setitem_34], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf96, buf95, buf92, buf93, buf94, buf97, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf98 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_2], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf91, buf90, buf97, buf98, 25214976, grid=grid(25214976), stream=stream0)
        buf99 = buf60; del buf60  # reuse
        buf100 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_8], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf98, buf99, buf100, 49152, 513, grid=grid(49152), stream=stream0)
        buf102 = reinterpret_tensor(buf85, (4096, 768), (768, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (4096, 768), (768, 1), 0), reinterpret_tensor(arg39_1, (768, 768), (1, 768), 0), out=buf102)
        del arg39_1
        buf103 = reinterpret_tensor(buf87, (48, 1536, 64), (98304, 64, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [padded_value_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf102, arg40_1, buf103, 4718592, grid=grid(4718592), stream=stream0)
        del arg40_1
        buf104 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf98, buf99, buf100, buf104, 37847040, grid=grid(37847040), stream=stream0)
        buf105 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [context_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf103, buf105, 9437184, grid=grid(9437184), stream=stream0)
        buf106 = reinterpret_tensor(buf102, (192, 256, 64), (16384, 64, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [context_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf105, (192, 768, 64), (49152, 64, 1), 0), out=buf106)
        buf107 = reinterpret_tensor(buf101, (4, 1024, 768), (786432, 768, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf106, buf107, 3145728, grid=grid(3145728), stream=stream0)
        buf108 = reinterpret_tensor(buf106, (4096, 768), (768, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (4096, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), out=buf108)
        del arg41_1
        buf112 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, add_13, hidden_states_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf108, arg42_1, buf79, arg43_1, arg44_1, buf112, 4096, 768, grid=grid(4096), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        buf113 = reinterpret_tensor(buf74, (4096, 3072), (3072, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (4096, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 3072), (1, 768), 0), out=buf113)
        del arg45_1
        buf114 = reinterpret_tensor(buf113, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf114, arg46_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg46_1
        buf115 = reinterpret_tensor(buf79, (4096, 768), (768, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg47_1, (3072, 768), (1, 3072), 0), out=buf115)
        del arg47_1
        buf119 = reinterpret_tensor(buf108, (4, 1024, 768), (786432, 768, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [add_14, hidden_states_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf115, arg48_1, buf112, arg49_1, arg50_1, buf119, 4096, 768, grid=grid(4096), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        buf120 = reinterpret_tensor(buf115, (1024, 4, 768), (3072, 768, 1), 0); del buf115  # reuse
        buf123 = reinterpret_tensor(buf112, (1024, 4, 768), (3072, 768, 1), 0); del buf112  # reuse
        buf141 = reinterpret_tensor(buf82, (1024, 4, 768), (3072, 768, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_9, key_vectors_6, value_vectors_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf119, buf120, buf123, buf141, 3145728, grid=grid(3145728), stream=stream0)
        buf121 = reinterpret_tensor(buf83, (4096, 768), (768, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (4096, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 768), (1, 768), 0), out=buf121)
        del arg51_1
        buf122 = reinterpret_tensor(buf121, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_6], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf122, arg52_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg52_1
        buf124 = reinterpret_tensor(buf120, (4096, 768), (768, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (4096, 768), (768, 1), 0), reinterpret_tensor(arg53_1, (768, 768), (1, 768), 0), out=buf124)
        del arg53_1
        buf125 = reinterpret_tensor(buf124, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf125, arg54_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg54_1
        buf126 = reinterpret_tensor(buf103, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf122, buf126, 4718592, grid=grid(4718592), stream=stream0)
        buf127 = reinterpret_tensor(buf86, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf125, buf127, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf128 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf127, (144, 64, 512), (32768, 512, 1), 0), out=buf128)
        buf129 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_12, setitem_36, setitem_37], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf128, buf129, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf130 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [setitem_38, setitem_39], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf128, buf129, buf130, 25214976, grid=grid(25214976), stream=stream0)
        buf131 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [bool_14, full_like_13, where_13, setitem_41], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf130, buf131, 6303744, grid=grid(6303744), stream=stream0)
        buf132 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf132, 4096, grid=grid(4096), stream=stream0)
        buf134 = reinterpret_tensor(buf97, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_14, setitem_42, setitem_43], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf132, buf133, buf134, 2101248, grid=grid(2101248), stream=stream0)
        buf135 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [setitem_45], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf132, buf133, buf134, buf135, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf136 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [bool_15, full_like_14, where_14], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf135, buf132, buf133, buf134, buf136, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf137 = reinterpret_tensor(buf94, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [setitem_46], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf136, buf135, buf132, buf133, buf134, buf137, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf138 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_3], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf131, buf130, buf137, buf138, 25214976, grid=grid(25214976), stream=stream0)
        buf139 = buf99; del buf99  # reuse
        buf140 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_12], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf138, buf139, buf140, 49152, 513, grid=grid(49152), stream=stream0)
        buf142 = reinterpret_tensor(buf125, (4096, 768), (768, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (4096, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 768), (1, 768), 0), out=buf142)
        del arg55_1
        buf143 = reinterpret_tensor(buf127, (48, 1536, 64), (98304, 64, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [padded_value_3], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf142, arg56_1, buf143, 4718592, grid=grid(4718592), stream=stream0)
        del arg56_1
        buf144 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_15], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf138, buf139, buf140, buf144, 37847040, grid=grid(37847040), stream=stream0)
        buf145 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [context_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf143, buf145, 9437184, grid=grid(9437184), stream=stream0)
        buf146 = reinterpret_tensor(buf142, (192, 256, 64), (16384, 64, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [context_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf145, (192, 768, 64), (49152, 64, 1), 0), out=buf146)
        buf147 = reinterpret_tensor(buf141, (4, 1024, 768), (786432, 768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf146, buf147, 3145728, grid=grid(3145728), stream=stream0)
        buf148 = reinterpret_tensor(buf146, (4096, 768), (768, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (4096, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), out=buf148)
        del arg57_1
        buf152 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_44, add_18, hidden_states_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf148, arg58_1, buf119, arg59_1, arg60_1, buf152, 4096, 768, grid=grid(4096), stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        buf153 = reinterpret_tensor(buf114, (4096, 3072), (3072, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf152, (4096, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 3072), (1, 768), 0), out=buf153)
        del arg61_1
        buf154 = reinterpret_tensor(buf153, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf154, arg62_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg62_1
        buf155 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg63_1, (3072, 768), (1, 3072), 0), out=buf155)
        del arg63_1
        buf159 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [add_19, hidden_states_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf155, arg64_1, buf152, arg65_1, arg66_1, buf159, 4096, 768, grid=grid(4096), stream=stream0)
        del arg64_1
        del arg65_1
        del arg66_1
        buf160 = reinterpret_tensor(buf155, (1024, 4, 768), (3072, 768, 1), 0); del buf155  # reuse
        buf163 = reinterpret_tensor(buf152, (1024, 4, 768), (3072, 768, 1), 0); del buf152  # reuse
        buf181 = reinterpret_tensor(buf122, (1024, 4, 768), (3072, 768, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_12, key_vectors_8, value_vectors_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf159, buf160, buf163, buf181, 3145728, grid=grid(3145728), stream=stream0)
        buf161 = reinterpret_tensor(buf123, (4096, 768), (768, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (4096, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 768), (1, 768), 0), out=buf161)
        del arg67_1
        buf162 = reinterpret_tensor(buf161, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf162, arg68_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg68_1
        buf164 = reinterpret_tensor(buf160, (4096, 768), (768, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (4096, 768), (768, 1), 0), reinterpret_tensor(arg69_1, (768, 768), (1, 768), 0), out=buf164)
        del arg69_1
        buf165 = reinterpret_tensor(buf164, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_54], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf165, arg70_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg70_1
        buf166 = reinterpret_tensor(buf143, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf162, buf166, 4718592, grid=grid(4718592), stream=stream0)
        buf167 = reinterpret_tensor(buf126, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf165, buf167, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf168 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf167, (144, 64, 512), (32768, 512, 1), 0), out=buf168)
        buf169 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_16, setitem_48, setitem_49], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf168, buf169, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf170 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [setitem_50, setitem_51], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf168, buf169, buf170, 25214976, grid=grid(25214976), stream=stream0)
        buf171 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [bool_18, full_like_17, where_17, setitem_53], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf170, buf171, 6303744, grid=grid(6303744), stream=stream0)
        buf172 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf172, 4096, grid=grid(4096), stream=stream0)
        buf173 = buf132; del buf132  # reuse
        buf213 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_69], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf173, buf213, 4096, grid=grid(4096), stream=stream0)
        buf174 = reinterpret_tensor(buf137, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_18, setitem_54, setitem_55], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf172, buf173, buf174, 2101248, grid=grid(2101248), stream=stream0)
        buf175 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [setitem_57], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf172, buf173, buf174, buf175, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf176 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [bool_19, full_like_18, where_18], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf175, buf172, buf173, buf174, buf176, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf177 = reinterpret_tensor(buf134, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [setitem_58], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf176, buf175, buf172, buf173, buf174, buf177, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf178 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_4], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf171, buf170, buf177, buf178, 25214976, grid=grid(25214976), stream=stream0)
        buf179 = buf140; del buf140  # reuse
        buf180 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_16], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf178, buf179, buf180, 49152, 513, grid=grid(49152), stream=stream0)
        buf182 = reinterpret_tensor(buf165, (4096, 768), (768, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (4096, 768), (768, 1), 0), reinterpret_tensor(arg71_1, (768, 768), (1, 768), 0), out=buf182)
        del arg71_1
        buf183 = reinterpret_tensor(buf167, (48, 1536, 64), (98304, 64, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [padded_value_4], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf182, arg72_1, buf183, 4718592, grid=grid(4718592), stream=stream0)
        del arg72_1
        buf184 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_20], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf178, buf179, buf180, buf184, 37847040, grid=grid(37847040), stream=stream0)
        buf185 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [context_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf183, buf185, 9437184, grid=grid(9437184), stream=stream0)
        buf186 = reinterpret_tensor(buf182, (192, 256, 64), (16384, 64, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [context_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf185, (192, 768, 64), (49152, 64, 1), 0), out=buf186)
        buf187 = reinterpret_tensor(buf181, (4, 1024, 768), (786432, 768, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf186, buf187, 3145728, grid=grid(3145728), stream=stream0)
        buf188 = reinterpret_tensor(buf186, (4096, 768), (768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (4096, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), out=buf188)
        del arg73_1
        buf192 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57, add_23, hidden_states_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf188, arg74_1, buf159, arg75_1, arg76_1, buf192, 4096, 768, grid=grid(4096), stream=stream0)
        del arg74_1
        del arg75_1
        del arg76_1
        buf193 = reinterpret_tensor(buf154, (4096, 3072), (3072, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf192, (4096, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 3072), (1, 768), 0), out=buf193)
        del arg77_1
        buf194 = reinterpret_tensor(buf193, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf194, arg78_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg78_1
        buf195 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf194, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg79_1, (3072, 768), (1, 3072), 0), out=buf195)
        del arg79_1
        buf199 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [add_24, hidden_states_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf195, arg80_1, buf192, arg81_1, arg82_1, buf199, 4096, 768, grid=grid(4096), stream=stream0)
        del arg80_1
        del arg81_1
        del arg82_1
        buf200 = reinterpret_tensor(buf195, (1024, 4, 768), (3072, 768, 1), 0); del buf195  # reuse
        buf203 = reinterpret_tensor(buf192, (1024, 4, 768), (3072, 768, 1), 0); del buf192  # reuse
        buf221 = reinterpret_tensor(buf162, (1024, 4, 768), (3072, 768, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_15, key_vectors_10, value_vectors_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf199, buf200, buf203, buf221, 3145728, grid=grid(3145728), stream=stream0)
        buf201 = reinterpret_tensor(buf163, (4096, 768), (768, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (4096, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 768), (1, 768), 0), out=buf201)
        del arg83_1
        buf202 = reinterpret_tensor(buf201, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_10], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf202, arg84_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg84_1
        buf204 = reinterpret_tensor(buf200, (4096, 768), (768, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (4096, 768), (768, 1), 0), reinterpret_tensor(arg85_1, (768, 768), (1, 768), 0), out=buf204)
        del arg85_1
        buf205 = reinterpret_tensor(buf204, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf205, arg86_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg86_1
        buf206 = reinterpret_tensor(buf183, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf202, buf206, 4718592, grid=grid(4718592), stream=stream0)
        buf207 = reinterpret_tensor(buf166, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf205, buf207, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf208 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf206, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf207, (144, 64, 512), (32768, 512, 1), 0), out=buf208)
        buf209 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_20, setitem_60, setitem_61], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf208, buf209, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf210 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [setitem_62, setitem_63], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf208, buf209, buf210, 25214976, grid=grid(25214976), stream=stream0)
        buf211 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [bool_22, full_like_21, where_21, setitem_65], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf210, buf211, 6303744, grid=grid(6303744), stream=stream0)
        buf212 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_68], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf212, 4096, grid=grid(4096), stream=stream0)
        buf214 = reinterpret_tensor(buf177, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_22, setitem_66, setitem_67], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf212, buf213, buf214, 2101248, grid=grid(2101248), stream=stream0)
        buf215 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [setitem_69], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf212, buf213, buf214, buf215, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf216 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [bool_23, full_like_22, where_22], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf215, buf212, buf213, buf214, buf216, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf217 = reinterpret_tensor(buf174, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [setitem_70], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf216, buf215, buf212, buf213, buf214, buf217, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf218 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_5], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf211, buf210, buf217, buf218, 25214976, grid=grid(25214976), stream=stream0)
        buf219 = buf180; del buf180  # reuse
        buf220 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_20], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf218, buf219, buf220, 49152, 513, grid=grid(49152), stream=stream0)
        buf222 = reinterpret_tensor(buf205, (4096, 768), (768, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (4096, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf222)
        del arg87_1
        buf223 = reinterpret_tensor(buf207, (48, 1536, 64), (98304, 64, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [padded_value_5], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf222, arg88_1, buf223, 4718592, grid=grid(4718592), stream=stream0)
        del arg88_1
        buf224 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_25], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf218, buf219, buf220, buf224, 37847040, grid=grid(37847040), stream=stream0)
        buf225 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [context_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf223, buf225, 9437184, grid=grid(9437184), stream=stream0)
        buf226 = reinterpret_tensor(buf222, (192, 256, 64), (16384, 64, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [context_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf225, (192, 768, 64), (49152, 64, 1), 0), out=buf226)
        buf227 = reinterpret_tensor(buf221, (4, 1024, 768), (786432, 768, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf226, buf227, 3145728, grid=grid(3145728), stream=stream0)
        buf228 = reinterpret_tensor(buf226, (4096, 768), (768, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (4096, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), out=buf228)
        del arg89_1
        buf232 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_70, add_28, hidden_states_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf228, arg90_1, buf199, arg91_1, arg92_1, buf232, 4096, 768, grid=grid(4096), stream=stream0)
        del arg90_1
        del arg91_1
        del arg92_1
        buf233 = reinterpret_tensor(buf194, (4096, 3072), (3072, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 3072), (1, 768), 0), out=buf233)
        del arg93_1
        buf234 = reinterpret_tensor(buf233, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_74], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf234, arg94_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg94_1
        buf235 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg95_1, (3072, 768), (1, 3072), 0), out=buf235)
        del arg95_1
        buf239 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [add_29, hidden_states_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf235, arg96_1, buf232, arg97_1, arg98_1, buf239, 4096, 768, grid=grid(4096), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf240 = reinterpret_tensor(buf235, (1024, 4, 768), (3072, 768, 1), 0); del buf235  # reuse
        buf243 = reinterpret_tensor(buf232, (1024, 4, 768), (3072, 768, 1), 0); del buf232  # reuse
        buf261 = reinterpret_tensor(buf202, (1024, 4, 768), (3072, 768, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_18, key_vectors_12, value_vectors_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf239, buf240, buf243, buf261, 3145728, grid=grid(3145728), stream=stream0)
        buf241 = reinterpret_tensor(buf203, (4096, 768), (768, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (4096, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 768), (1, 768), 0), out=buf241)
        del arg99_1
        buf242 = reinterpret_tensor(buf241, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf242, arg100_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg100_1
        buf244 = reinterpret_tensor(buf240, (4096, 768), (768, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (4096, 768), (768, 1), 0), reinterpret_tensor(arg101_1, (768, 768), (1, 768), 0), out=buf244)
        del arg101_1
        buf245 = reinterpret_tensor(buf244, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_80], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf245, arg102_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg102_1
        buf246 = reinterpret_tensor(buf223, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf242, buf246, 4718592, grid=grid(4718592), stream=stream0)
        buf247 = reinterpret_tensor(buf206, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf245, buf247, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf248 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf247, (144, 64, 512), (32768, 512, 1), 0), out=buf248)
        buf249 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_24, setitem_72, setitem_73], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf248, buf249, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf250 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [setitem_74, setitem_75], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf248, buf249, buf250, 25214976, grid=grid(25214976), stream=stream0)
        buf251 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [bool_26, full_like_25, where_25, setitem_77], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf250, buf251, 6303744, grid=grid(6303744), stream=stream0)
        buf252 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf252, 4096, grid=grid(4096), stream=stream0)
        buf253 = buf212; del buf212  # reuse
        buf293 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_95], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf253, buf293, 4096, grid=grid(4096), stream=stream0)
        buf254 = reinterpret_tensor(buf217, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_26, setitem_78, setitem_79], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf252, buf253, buf254, 2101248, grid=grid(2101248), stream=stream0)
        buf255 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [setitem_81], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf252, buf253, buf254, buf255, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf256 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [bool_27, full_like_26, where_26], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf255, buf252, buf253, buf254, buf256, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf257 = reinterpret_tensor(buf214, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [setitem_82], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf256, buf255, buf252, buf253, buf254, buf257, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf258 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_6], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf251, buf250, buf257, buf258, 25214976, grid=grid(25214976), stream=stream0)
        buf259 = buf220; del buf220  # reuse
        buf260 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_24], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf258, buf259, buf260, 49152, 513, grid=grid(49152), stream=stream0)
        buf262 = reinterpret_tensor(buf245, (4096, 768), (768, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (4096, 768), (768, 1), 0), reinterpret_tensor(arg103_1, (768, 768), (1, 768), 0), out=buf262)
        del arg103_1
        buf263 = reinterpret_tensor(buf247, (48, 1536, 64), (98304, 64, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [padded_value_6], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf262, arg104_1, buf263, 4718592, grid=grid(4718592), stream=stream0)
        del arg104_1
        buf264 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_30], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf258, buf259, buf260, buf264, 37847040, grid=grid(37847040), stream=stream0)
        buf265 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [context_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf263, buf265, 9437184, grid=grid(9437184), stream=stream0)
        buf266 = reinterpret_tensor(buf262, (192, 256, 64), (16384, 64, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [context_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf264, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf265, (192, 768, 64), (49152, 64, 1), 0), out=buf266)
        buf267 = reinterpret_tensor(buf261, (4, 1024, 768), (786432, 768, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf266, buf267, 3145728, grid=grid(3145728), stream=stream0)
        buf268 = reinterpret_tensor(buf266, (4096, 768), (768, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (4096, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), out=buf268)
        del arg105_1
        buf272 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83, add_33, hidden_states_85], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf268, arg106_1, buf239, arg107_1, arg108_1, buf272, 4096, 768, grid=grid(4096), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        buf273 = reinterpret_tensor(buf234, (4096, 3072), (3072, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf272, (4096, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 3072), (1, 768), 0), out=buf273)
        del arg109_1
        buf274 = reinterpret_tensor(buf273, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf274, arg110_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg110_1
        buf275 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf274, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg111_1, (3072, 768), (1, 3072), 0), out=buf275)
        del arg111_1
        buf279 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [add_34, hidden_states_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf275, arg112_1, buf272, arg113_1, arg114_1, buf279, 4096, 768, grid=grid(4096), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        buf280 = reinterpret_tensor(buf275, (1024, 4, 768), (3072, 768, 1), 0); del buf275  # reuse
        buf283 = reinterpret_tensor(buf272, (1024, 4, 768), (3072, 768, 1), 0); del buf272  # reuse
        buf301 = reinterpret_tensor(buf242, (1024, 4, 768), (3072, 768, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_21, key_vectors_14, value_vectors_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf279, buf280, buf283, buf301, 3145728, grid=grid(3145728), stream=stream0)
        buf281 = reinterpret_tensor(buf243, (4096, 768), (768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (4096, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 768), (1, 768), 0), out=buf281)
        del arg115_1
        buf282 = reinterpret_tensor(buf281, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_14], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf282, arg116_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg116_1
        buf284 = reinterpret_tensor(buf280, (4096, 768), (768, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (4096, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf284)
        del arg117_1
        buf285 = reinterpret_tensor(buf284, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf285, arg118_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg118_1
        buf286 = reinterpret_tensor(buf263, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf282, buf286, 4718592, grid=grid(4718592), stream=stream0)
        buf287 = reinterpret_tensor(buf246, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf285, buf287, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf288 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf286, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf287, (144, 64, 512), (32768, 512, 1), 0), out=buf288)
        buf289 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_28, setitem_84, setitem_85], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf288, buf289, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf290 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [setitem_86, setitem_87], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf288, buf289, buf290, 25214976, grid=grid(25214976), stream=stream0)
        buf291 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [bool_30, full_like_29, where_29, setitem_89], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf290, buf291, 6303744, grid=grid(6303744), stream=stream0)
        buf292 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf292, 4096, grid=grid(4096), stream=stream0)
        buf294 = reinterpret_tensor(buf257, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_30, setitem_90, setitem_91], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf292, buf293, buf294, 2101248, grid=grid(2101248), stream=stream0)
        buf295 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [setitem_93], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf292, buf293, buf294, buf295, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf296 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [bool_31, full_like_30, where_30], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf295, buf292, buf293, buf294, buf296, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf297 = reinterpret_tensor(buf254, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [setitem_94], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf296, buf295, buf292, buf293, buf294, buf297, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf298 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_7], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf291, buf290, buf297, buf298, 25214976, grid=grid(25214976), stream=stream0)
        buf299 = buf260; del buf260  # reuse
        buf300 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_28], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf298, buf299, buf300, 49152, 513, grid=grid(49152), stream=stream0)
        buf302 = reinterpret_tensor(buf285, (4096, 768), (768, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (4096, 768), (768, 1), 0), reinterpret_tensor(arg119_1, (768, 768), (1, 768), 0), out=buf302)
        del arg119_1
        buf303 = reinterpret_tensor(buf287, (48, 1536, 64), (98304, 64, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [padded_value_7], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf302, arg120_1, buf303, 4718592, grid=grid(4718592), stream=stream0)
        del arg120_1
        buf304 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_35], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf298, buf299, buf300, buf304, 37847040, grid=grid(37847040), stream=stream0)
        buf305 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [context_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf303, buf305, 9437184, grid=grid(9437184), stream=stream0)
        buf306 = reinterpret_tensor(buf302, (192, 256, 64), (16384, 64, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [context_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf305, (192, 768, 64), (49152, 64, 1), 0), out=buf306)
        buf307 = reinterpret_tensor(buf301, (4, 1024, 768), (786432, 768, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf306, buf307, 3145728, grid=grid(3145728), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (4096, 768), (768, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (4096, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), out=buf308)
        del arg121_1
        buf312 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96, add_38, hidden_states_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf308, arg122_1, buf279, arg123_1, arg124_1, buf312, 4096, 768, grid=grid(4096), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        buf313 = reinterpret_tensor(buf274, (4096, 3072), (3072, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (4096, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 3072), (1, 768), 0), out=buf313)
        del arg125_1
        buf314 = reinterpret_tensor(buf313, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf314, arg126_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg126_1
        buf315 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf314, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg127_1, (3072, 768), (1, 3072), 0), out=buf315)
        del arg127_1
        buf319 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [add_39, hidden_states_103], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf315, arg128_1, buf312, arg129_1, arg130_1, buf319, 4096, 768, grid=grid(4096), stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        buf320 = reinterpret_tensor(buf315, (1024, 4, 768), (3072, 768, 1), 0); del buf315  # reuse
        buf323 = reinterpret_tensor(buf312, (1024, 4, 768), (3072, 768, 1), 0); del buf312  # reuse
        buf341 = reinterpret_tensor(buf282, (1024, 4, 768), (3072, 768, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_24, key_vectors_16, value_vectors_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf319, buf320, buf323, buf341, 3145728, grid=grid(3145728), stream=stream0)
        buf321 = reinterpret_tensor(buf283, (4096, 768), (768, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (4096, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 768), (1, 768), 0), out=buf321)
        del arg131_1
        buf322 = reinterpret_tensor(buf321, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf322, arg132_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg132_1
        buf324 = reinterpret_tensor(buf320, (4096, 768), (768, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (4096, 768), (768, 1), 0), reinterpret_tensor(arg133_1, (768, 768), (1, 768), 0), out=buf324)
        del arg133_1
        buf325 = reinterpret_tensor(buf324, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_106], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf325, arg134_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg134_1
        buf326 = reinterpret_tensor(buf303, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf322, buf326, 4718592, grid=grid(4718592), stream=stream0)
        buf327 = reinterpret_tensor(buf286, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf325, buf327, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf328 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf326, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf327, (144, 64, 512), (32768, 512, 1), 0), out=buf328)
        buf329 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_32, setitem_96, setitem_97], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf328, buf329, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf330 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [setitem_98, setitem_99], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf328, buf329, buf330, 25214976, grid=grid(25214976), stream=stream0)
        buf331 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [bool_34, full_like_33, where_33, setitem_101], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf330, buf331, 6303744, grid=grid(6303744), stream=stream0)
        buf332 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf332, 4096, grid=grid(4096), stream=stream0)
        buf333 = buf292; del buf292  # reuse
        buf373 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_121], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf333, buf373, 4096, grid=grid(4096), stream=stream0)
        buf334 = reinterpret_tensor(buf297, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_34, setitem_102, setitem_103], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf332, buf333, buf334, 2101248, grid=grid(2101248), stream=stream0)
        buf335 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [setitem_105], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf332, buf333, buf334, buf335, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf336 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [bool_35, full_like_34, where_34], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf335, buf332, buf333, buf334, buf336, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf337 = reinterpret_tensor(buf294, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [setitem_106], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf336, buf335, buf332, buf333, buf334, buf337, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf338 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_8], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf331, buf330, buf337, buf338, 25214976, grid=grid(25214976), stream=stream0)
        buf339 = buf300; del buf300  # reuse
        buf340 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_32], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf338, buf339, buf340, 49152, 513, grid=grid(49152), stream=stream0)
        buf342 = reinterpret_tensor(buf325, (4096, 768), (768, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (4096, 768), (768, 1), 0), reinterpret_tensor(arg135_1, (768, 768), (1, 768), 0), out=buf342)
        del arg135_1
        buf343 = reinterpret_tensor(buf327, (48, 1536, 64), (98304, 64, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [padded_value_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf342, arg136_1, buf343, 4718592, grid=grid(4718592), stream=stream0)
        del arg136_1
        buf344 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_40], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf338, buf339, buf340, buf344, 37847040, grid=grid(37847040), stream=stream0)
        buf345 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [context_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf343, buf345, 9437184, grid=grid(9437184), stream=stream0)
        buf346 = reinterpret_tensor(buf342, (192, 256, 64), (16384, 64, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [context_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf344, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf345, (192, 768, 64), (49152, 64, 1), 0), out=buf346)
        buf347 = reinterpret_tensor(buf341, (4, 1024, 768), (786432, 768, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf346, buf347, 3145728, grid=grid(3145728), stream=stream0)
        buf348 = reinterpret_tensor(buf346, (4096, 768), (768, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (4096, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf348)
        del arg137_1
        buf352 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109, add_43, hidden_states_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf348, arg138_1, buf319, arg139_1, arg140_1, buf352, 4096, 768, grid=grid(4096), stream=stream0)
        del arg138_1
        del arg139_1
        del arg140_1
        buf353 = reinterpret_tensor(buf314, (4096, 3072), (3072, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf352, (4096, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 3072), (1, 768), 0), out=buf353)
        del arg141_1
        buf354 = reinterpret_tensor(buf353, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_113], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf354, arg142_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg142_1
        buf355 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg143_1, (3072, 768), (1, 3072), 0), out=buf355)
        del arg143_1
        buf359 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [add_44, hidden_states_116], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf355, arg144_1, buf352, arg145_1, arg146_1, buf359, 4096, 768, grid=grid(4096), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        buf360 = reinterpret_tensor(buf355, (1024, 4, 768), (3072, 768, 1), 0); del buf355  # reuse
        buf363 = reinterpret_tensor(buf352, (1024, 4, 768), (3072, 768, 1), 0); del buf352  # reuse
        buf381 = reinterpret_tensor(buf322, (1024, 4, 768), (3072, 768, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_27, key_vectors_18, value_vectors_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf359, buf360, buf363, buf381, 3145728, grid=grid(3145728), stream=stream0)
        buf361 = reinterpret_tensor(buf323, (4096, 768), (768, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (4096, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf361)
        del arg147_1
        buf362 = reinterpret_tensor(buf361, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_18], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf362, arg148_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg148_1
        buf364 = reinterpret_tensor(buf360, (4096, 768), (768, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (4096, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 768), (1, 768), 0), out=buf364)
        del arg149_1
        buf365 = reinterpret_tensor(buf364, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_119], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf365, arg150_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg150_1
        buf366 = reinterpret_tensor(buf343, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf362, buf366, 4718592, grid=grid(4718592), stream=stream0)
        buf367 = reinterpret_tensor(buf326, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf365, buf367, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf368 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf366, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf367, (144, 64, 512), (32768, 512, 1), 0), out=buf368)
        buf369 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_36, setitem_108, setitem_109], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf368, buf369, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf370 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [setitem_110, setitem_111], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf368, buf369, buf370, 25214976, grid=grid(25214976), stream=stream0)
        buf371 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [bool_38, full_like_37, where_37, setitem_113], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf370, buf371, 6303744, grid=grid(6303744), stream=stream0)
        buf372 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_120], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf372, 4096, grid=grid(4096), stream=stream0)
        buf374 = reinterpret_tensor(buf337, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_38, setitem_114, setitem_115], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf372, buf373, buf374, 2101248, grid=grid(2101248), stream=stream0)
        buf375 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [setitem_117], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf372, buf373, buf374, buf375, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf376 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [bool_39, full_like_38, where_38], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf375, buf372, buf373, buf374, buf376, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf377 = reinterpret_tensor(buf334, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [setitem_118], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf376, buf375, buf372, buf373, buf374, buf377, 4096, 513, grid=grid(4096, 513), stream=stream0)
        buf378 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_9], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf371, buf370, buf377, buf378, 25214976, grid=grid(25214976), stream=stream0)
        buf379 = buf340; del buf340  # reuse
        buf380 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_36], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf378, buf379, buf380, 49152, 513, grid=grid(49152), stream=stream0)
        buf382 = reinterpret_tensor(buf365, (4096, 768), (768, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (4096, 768), (768, 1), 0), reinterpret_tensor(arg151_1, (768, 768), (1, 768), 0), out=buf382)
        del arg151_1
        buf383 = reinterpret_tensor(buf367, (48, 1536, 64), (98304, 64, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [padded_value_9], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf382, arg152_1, buf383, 4718592, grid=grid(4718592), stream=stream0)
        del arg152_1
        buf384 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_45], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf378, buf379, buf380, buf384, 37847040, grid=grid(37847040), stream=stream0)
        buf385 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [context_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf383, buf385, 9437184, grid=grid(9437184), stream=stream0)
        buf386 = reinterpret_tensor(buf382, (192, 256, 64), (16384, 64, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [context_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf385, (192, 768, 64), (49152, 64, 1), 0), out=buf386)
        buf387 = reinterpret_tensor(buf381, (4, 1024, 768), (786432, 768, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_122], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf386, buf387, 3145728, grid=grid(3145728), stream=stream0)
        buf388 = reinterpret_tensor(buf386, (4096, 768), (768, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_122], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (4096, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), out=buf388)
        del arg153_1
        buf392 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_122, add_48, hidden_states_124], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf388, arg154_1, buf359, arg155_1, arg156_1, buf392, 4096, 768, grid=grid(4096), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        buf393 = reinterpret_tensor(buf354, (4096, 3072), (3072, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (4096, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 3072), (1, 768), 0), out=buf393)
        del arg157_1
        buf394 = reinterpret_tensor(buf393, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf394, arg158_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg158_1
        buf395 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg159_1, (3072, 768), (1, 3072), 0), out=buf395)
        del arg159_1
        buf399 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [add_49, hidden_states_129], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf395, arg160_1, buf392, arg161_1, arg162_1, buf399, 4096, 768, grid=grid(4096), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        buf400 = reinterpret_tensor(buf395, (1024, 4, 768), (3072, 768, 1), 0); del buf395  # reuse
        buf403 = reinterpret_tensor(buf392, (1024, 4, 768), (3072, 768, 1), 0); del buf392  # reuse
        buf421 = reinterpret_tensor(buf362, (1024, 4, 768), (3072, 768, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_30, key_vectors_20, value_vectors_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf399, buf400, buf403, buf421, 3145728, grid=grid(3145728), stream=stream0)
        buf401 = reinterpret_tensor(buf363, (4096, 768), (768, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (4096, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 768), (1, 768), 0), out=buf401)
        del arg163_1
        buf402 = reinterpret_tensor(buf401, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf402, arg164_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg164_1
        buf404 = reinterpret_tensor(buf400, (4096, 768), (768, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (4096, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), out=buf404)
        del arg165_1
        buf405 = reinterpret_tensor(buf404, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_132], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf405, arg166_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg166_1
        buf406 = reinterpret_tensor(buf383, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf402, buf406, 4718592, grid=grid(4718592), stream=stream0)
        buf407 = reinterpret_tensor(buf366, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf405, buf407, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf408 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf407, (144, 64, 512), (32768, 512, 1), 0), out=buf408)
        buf409 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_40, setitem_120, setitem_121], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf408, buf409, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf410 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [setitem_122, setitem_123], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf408, buf409, buf410, 25214976, grid=grid(25214976), stream=stream0)
        buf411 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [bool_42, full_like_41, where_41, setitem_125], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf410, buf411, 6303744, grid=grid(6303744), stream=stream0)
        buf412 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf412, 4096, grid=grid(4096), stream=stream0)
        buf413 = buf372; del buf372  # reuse
        buf453 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_134, hidden_states_147], Original ATen: [aten.view]
        triton_poi_fused_view_9.run(arg7_1, buf413, buf453, 4096, grid=grid(4096), stream=stream0)
        del arg7_1
        buf414 = reinterpret_tensor(buf377, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf377  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_42, setitem_126, setitem_127], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf412, buf413, buf414, 2101248, grid=grid(2101248), stream=stream0)
        buf415 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [setitem_129], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf412, buf413, buf414, buf415, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf416 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [bool_43, full_like_42, where_42], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf415, buf412, buf413, buf414, buf416, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf417 = reinterpret_tensor(buf374, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [setitem_130], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf416, buf415, buf412, buf413, buf414, buf417, 4096, 513, grid=grid(4096, 513), stream=stream0)
        del buf412
        buf418 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_10], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf411, buf410, buf417, buf418, 25214976, grid=grid(25214976), stream=stream0)
        buf419 = buf380; del buf380  # reuse
        buf420 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_40], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf418, buf419, buf420, 49152, 513, grid=grid(49152), stream=stream0)
        buf422 = reinterpret_tensor(buf405, (4096, 768), (768, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (4096, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 768), (1, 768), 0), out=buf422)
        del arg167_1
        buf423 = reinterpret_tensor(buf407, (48, 1536, 64), (98304, 64, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [padded_value_10], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf422, arg168_1, buf423, 4718592, grid=grid(4718592), stream=stream0)
        del arg168_1
        buf424 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_50], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf418, buf419, buf420, buf424, 37847040, grid=grid(37847040), stream=stream0)
        buf425 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [context_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf423, buf425, 9437184, grid=grid(9437184), stream=stream0)
        buf426 = reinterpret_tensor(buf422, (192, 256, 64), (16384, 64, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [context_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf425, (192, 768, 64), (49152, 64, 1), 0), out=buf426)
        buf427 = reinterpret_tensor(buf421, (4, 1024, 768), (786432, 768, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf426, buf427, 3145728, grid=grid(3145728), stream=stream0)
        buf428 = reinterpret_tensor(buf426, (4096, 768), (768, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (4096, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), out=buf428)
        del arg169_1
        buf432 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135, add_53, hidden_states_137], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf428, arg170_1, buf399, arg171_1, arg172_1, buf432, 4096, 768, grid=grid(4096), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        buf433 = reinterpret_tensor(buf394, (4096, 3072), (3072, 1), 0); del buf394  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf432, (4096, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 3072), (1, 768), 0), out=buf433)
        del arg173_1
        buf434 = reinterpret_tensor(buf433, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf434, arg174_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg174_1
        buf435 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg175_1, (3072, 768), (1, 3072), 0), out=buf435)
        del arg175_1
        buf439 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [add_54, hidden_states_142], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf435, arg176_1, buf432, arg177_1, arg178_1, buf439, 4096, 768, grid=grid(4096), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        buf440 = reinterpret_tensor(buf435, (1024, 4, 768), (3072, 768, 1), 0); del buf435  # reuse
        buf443 = reinterpret_tensor(buf432, (1024, 4, 768), (3072, 768, 1), 0); del buf432  # reuse
        buf461 = reinterpret_tensor(buf402, (1024, 4, 768), (3072, 768, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_33, key_vectors_22, value_vectors_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_0.run(buf439, buf440, buf443, buf461, 3145728, grid=grid(3145728), stream=stream0)
        buf441 = reinterpret_tensor(buf403, (4096, 768), (768, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [query_vectors_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf440, (4096, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 768), (1, 768), 0), out=buf441)
        del arg179_1
        buf442 = reinterpret_tensor(buf441, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_22], Original ATen: [aten.view]
        triton_poi_fused_view_1.run(buf442, arg180_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg180_1
        buf444 = reinterpret_tensor(buf440, (4096, 768), (768, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [key_vectors_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (4096, 768), (768, 1), 0), reinterpret_tensor(arg181_1, (768, 768), (1, 768), 0), out=buf444)
        del arg181_1
        del buf443
        buf445 = reinterpret_tensor(buf444, (48, 2, 512, 64), (64, 1572864, 3072, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_145], Original ATen: [aten.view]
        triton_poi_fused_view_2.run(buf445, arg182_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg182_1
        buf446 = reinterpret_tensor(buf423, (48, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf442, buf446, 4718592, grid=grid(4718592), stream=stream0)
        del buf442
        buf447 = reinterpret_tensor(buf406, (48, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf445, buf447, 9216, 512, grid=grid(9216, 512), stream=stream0)
        buf448 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [diagonal_chunked_attention_scores_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (144, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf447, (144, 64, 512), (32768, 512, 1), 0), out=buf448)
        del buf446
        buf449 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_44, setitem_132, setitem_133], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_5.run(buf448, buf449, 256, 98496, grid=grid(256, 98496), stream=stream0)
        buf450 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [setitem_134, setitem_135], Original ATen: [aten.copy]
        triton_poi_fused_copy_6.run(buf448, buf449, buf450, 25214976, grid=grid(25214976), stream=stream0)
        del buf448
        del buf449
        buf451 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [bool_46, full_like_45, where_45, setitem_137], Original ATen: [aten._to_copy, aten.full_like, aten.where, aten.copy]
        triton_poi_fused__to_copy_copy_full_like_where_7.run(buf450, buf451, 6303744, grid=grid(6303744), stream=stream0)
        buf452 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_146], Original ATen: [aten.view]
        triton_poi_fused_view_8.run(buf452, 4096, grid=grid(4096), stream=stream0)
        buf454 = reinterpret_tensor(buf417, (4, 4, 256, 513), (131328, 525312, 513, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [diagonal_attention_scores_46, setitem_138, setitem_139], Original ATen: [aten.new_zeros, aten.copy]
        triton_poi_fused_copy_new_zeros_10.run(buf452, buf453, buf454, 2101248, grid=grid(2101248), stream=stream0)
        buf455 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [setitem_141], Original ATen: [aten.copy]
        triton_poi_fused_copy_11.run(buf452, buf453, buf454, buf455, 1024, 513, grid=grid(1024, 513), stream=stream0)
        buf456 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [bool_47, full_like_46, where_46], Original ATen: [aten._to_copy, aten.full_like, aten.where]
        triton_poi_fused__to_copy_full_like_where_12.run(buf455, buf452, buf453, buf454, buf456, 1024, 257, grid=grid(1024, 257), stream=stream0)
        buf457 = reinterpret_tensor(buf414, (4, 1024, 1, 513), (525312, 513, 2101248, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [setitem_142], Original ATen: [aten.copy]
        triton_poi_fused_copy_13.run(buf456, buf455, buf452, buf453, buf454, buf457, 4096, 513, grid=grid(4096, 513), stream=stream0)
        del buf452
        del buf453
        del buf454
        del buf455
        del buf456
        buf458 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_11], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf451, buf450, buf457, buf458, 25214976, grid=grid(25214976), stream=stream0)
        del buf450
        del buf451
        del buf457
        buf459 = buf420; del buf420  # reuse
        buf460 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [attn_probs_44], Original ATen: [aten._softmax]
        triton_red_fused__softmax_15.run(buf458, buf459, buf460, 49152, 513, grid=grid(49152), stream=stream0)
        buf462 = reinterpret_tensor(buf445, (4096, 768), (768, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [value_vectors_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (4096, 768), (768, 1), 0), reinterpret_tensor(arg183_1, (768, 768), (1, 768), 0), out=buf462)
        del arg183_1
        buf463 = reinterpret_tensor(buf447, (48, 1536, 64), (98304, 64, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [padded_value_11], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_16.run(buf462, arg184_1, buf463, 4718592, grid=grid(4718592), stream=stream0)
        del arg184_1
        buf464 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [chunked_hidden_states_55], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(arg8_1, buf458, buf459, buf460, buf464, 37847040, grid=grid(37847040), stream=stream0)
        del arg8_1
        del buf458
        del buf459
        del buf460
        buf465 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [context_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_18.run(buf463, buf465, 9437184, grid=grid(9437184), stream=stream0)
        del buf463
        buf466 = reinterpret_tensor(buf462, (192, 256, 64), (16384, 64, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [context_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf464, (192, 256, 768), (197120, 769, 1), 0), reinterpret_tensor(buf465, (192, 768, 64), (49152, 64, 1), 0), out=buf466)
        del buf464
        del buf465
        buf467 = reinterpret_tensor(buf461, (4, 1024, 768), (786432, 768, 1), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf466, buf467, 3145728, grid=grid(3145728), stream=stream0)
        buf468 = reinterpret_tensor(buf466, (4096, 768), (768, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf467, (4096, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), out=buf468)
        del arg185_1
        buf472 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148, add_58, hidden_states_150], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf468, arg186_1, buf439, arg187_1, arg188_1, buf472, 4096, 768, grid=grid(4096), stream=stream0)
        del arg186_1
        del arg187_1
        del arg188_1
        buf473 = reinterpret_tensor(buf434, (4096, 3072), (3072, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf472, (4096, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 3072), (1, 768), 0), out=buf473)
        del arg189_1
        buf474 = reinterpret_tensor(buf473, (4, 1024, 3072), (3145728, 3072, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_152], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_21.run(buf474, arg190_1, 12582912, grid=grid(12582912), stream=stream0)
        del arg190_1
        buf475 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg191_1, (3072, 768), (1, 3072), 0), out=buf475)
        del arg191_1
        del buf474
        buf479 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [add_59, hidden_states_155], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_20.run(buf475, arg192_1, buf472, arg193_1, arg194_1, buf479, 4096, 768, grid=grid(4096), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del buf472
        del buf475
    return (buf479, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.bool)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg99_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
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
    arg141_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
