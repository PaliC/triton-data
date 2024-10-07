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


# kernel path: /tmp/torchinductor_sahanp/vo/cvoyy34r3ojvxt444drapeq4zg5m6vf7eioef5fudv3zjt3gxrv6.py
# Topologically Sorted Source Nodes: [inputs_embeds_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   inputs_embeds_1 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%constant_pad_nd, %embedding, %constant_pad_nd_1], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 384
    x1 = (xindex // 384) % 128
    x3 = (xindex // 384)
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 127, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + (1 + x3), tmp8, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full([XBLOCK], 30522, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tl.broadcast_to(tmp13, [XBLOCK])) & (tl.broadcast_to(tmp13, [XBLOCK]) < 30522)) | ~(tmp8), "index out of bounds: 0 <= tl.broadcast_to(tmp13, [XBLOCK]) < 30522")
    tmp15 = tl.load(in_ptr1 + ((128*tmp13) + x0), tmp8, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr0 + (x3), tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 + tmp10
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tl.device_assert(((0 <= tl.broadcast_to(tmp25, [XBLOCK])) & (tl.broadcast_to(tmp25, [XBLOCK]) < 30522)) | ~(tmp21), "index out of bounds: 0 <= tl.broadcast_to(tmp25, [XBLOCK]) < 30522")
    tmp27 = tl.load(in_ptr1 + ((128*tmp25) + ((-128) + x0)), tmp21, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp0 >= tmp19
    tmp29 = tl.full([1], 384, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = (-1) + x1
    tmp32 = tmp31 >= tmp1
    tmp33 = tmp32 & tmp28
    tmp34 = tl.load(in_ptr0 + ((-1) + x3), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp10
    tmp36 = tmp34 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp34)
    tl.device_assert(((0 <= tl.broadcast_to(tmp37, [XBLOCK])) & (tl.broadcast_to(tmp37, [XBLOCK]) < 30522)) | ~(tmp33), "index out of bounds: 0 <= tl.broadcast_to(tmp37, [XBLOCK]) < 30522")
    tmp39 = tl.load(in_ptr1 + ((128*tmp37) + ((-256) + x0)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp28, tmp39, tmp40)
    tmp42 = tl.where(tmp21, tmp27, tmp41)
    tmp43 = tl.where(tmp4, tmp17, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3m/c3mengdaxs66lhywzkgbir32gwjefeum4hdfp2m55z6mnyzurtru.py
# Topologically Sorted Source Nodes: [position_embeddings, add, token_type_ids, token_type_embeddings, embeddings, mul_1, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.zeros, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   embeddings => add_1
#   embeddings_1 => add_2
#   mul_1 => mul_1
#   position_embeddings => embedding_1
#   token_type_embeddings => embedding_2
#   token_type_ids => full_default
# Graph fragment:
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %slice_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %embedding_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %embedding_2 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %full_default), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %arg7_1), kwargs = {})
#   %add_2 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg6_1), kwargs = {})
triton_poi_fused_add_embedding_mul_zeros_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_zeros_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK], 512, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert((0 <= tmp7) & (tmp7 < 512), "index out of bounds: 0 <= tmp7 < 512")
    tmp9 = tl.load(in_ptr2 + (x0 + (512*tmp7)), None)
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/am/camjfzs5ayykgqajopxqxvm2qs3uvjyz3ljjv4vuxnpk2eda6psm.py
# Topologically Sorted Source Nodes: [mul_3, layer_input_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   layer_input_3 => add_4
#   mul_3 => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %arg35_1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg34_1), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dx/cdxqr4hpjrwdevdfuls6sz7ipbr6lwzjgpknqcvywfyj5ubizabi.py
# Topologically Sorted Source Nodes: [mul_2, layer_input_1, add_6, mul_4, layer_outputs_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_6
#   layer_input_1 => add_3
#   layer_outputs_1 => add_7
#   mul_2 => mul_2
#   mul_4 => mul_4
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %arg31_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg30_1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_23, %add_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %arg17_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg16_1), kwargs = {})
triton_poi_fused_add_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(in_out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ye/cyesiojlunwnbp4um4dxzcxho3kxawjglkf7lrf6ejrf7gnq46z3.py
# Topologically Sorted Source Nodes: [hidden_states_1], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   hidden_states_1 => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_25,), kwargs = {})
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
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4r/c4raeig7ids6laemvogwmf5sqv5o3laherl2kzr3jumlz4huim7y.py
# Topologically Sorted Source Nodes: [add_8, mul_5, layer_outputs_3], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_8 => add_8
#   layer_outputs_3 => add_9
#   mul_5 => mul_5
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_27, %add_7), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %arg41_1), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg40_1), kwargs = {})
triton_poi_fused_add_mul_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/id/cid3lmvccb3ga5pw5veu2ttad5z47lvkdyeb3ugbau3uswomvzsu.py
# Topologically Sorted Source Nodes: [add_16, mul_9, layer_outputs_10], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_16 => add_16
#   layer_outputs_10 => add_17
#   mul_9 => mul_9
# Graph fragment:
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_41, %add_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %arg27_1), kwargs = {})
#   %add_17 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg26_1), kwargs = {})
triton_poi_fused_add_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mi/cmixyo2xka3ltwzr3x344rwiutyy4hmaxoccf2zmnwhns57rfuun.py
# Topologically Sorted Source Nodes: [add_61, mul_33, layer_outputs_43], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_61 => add_61
#   layer_outputs_43 => add_62
#   mul_33 => mul_33
# Graph fragment:
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_161, %add_47), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, %arg165_1), kwargs = {})
#   %add_62 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg164_1), kwargs = {})
triton_poi_fused_add_mul_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/k2/ck2pcm3iuzncr6w46vk4adnzwwknxrceey7ealzkg5n2wbgjevqi.py
# Topologically Sorted Source Nodes: [hidden_states_193, hidden_states_194], Original ATen: [aten.relu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_193 => relu_96
#   hidden_states_194 => add_363, add_364, mul_194, mul_195, rsqrt, sub_25, var_mean
# Graph fragment:
#   %relu_96 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_963,), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%relu_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_96, %getitem_1), kwargs = {})
#   %add_363 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_363,), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_194, %arg1115_1), kwargs = {})
#   %add_364 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %arg1116_1), kwargs = {})
triton_per_fused_native_layer_norm_relu_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_relu_8', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 16384
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
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
    tmp21 = 1e-12
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/26/c26ef7wlfl5uoegblup5w74p4avrurzke7vzurc5nkd2omclaki2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_default : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_1, %full_default_4], 1), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15628288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 30524
    x1 = (xindex // 30524)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 30522, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp8 & tmp4
    tmp10 = tl.load(in_ptr0 + ((128*x0) + x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp5 >= tmp7
    tmp12 = tl.full([1], 512, tl.int64)
    tmp13 = tmp5 < tmp12
    tmp14 = tmp11 & tmp4
    tmp15 = tl.load(in_ptr1 + ((30522*((-128) + x1)) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp8, tmp10, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 30524, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = 0.0
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp4, tmp18, tmp24)
    tl.store(out_ptr0 + (x0 + (30528*x1)), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hw/chwuo4rkqzmae3y3lhpwrezol6zc5usc7wtjrpb46ltlimatkb6w.py
# Topologically Sorted Source Nodes: [hidden_states_196, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
# Source node to ATen node mapping:
#   hidden_states_196 => add_365
#   masked_lm_loss => amax_24, exp_24, sub_26, sum_25
# Graph fragment:
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_965, %arg1118_1), kwargs = {})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_971, [1], True), kwargs = {})
#   %sub_26 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_971, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_26,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_add_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 30522
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
        tmp0 = tl.load(in_ptr0 + (r1 + (30528*x0) + (3907584*((x0 % 128) // 128))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (30528*x0) + (3907584*((x0 % 128) // 128))), rmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr0 + (r1 + (30528*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8 - tmp4
        tmp10 = tl_math.exp(tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp15 = tmp14 + tmp7
        tl.store(out_ptr2 + (r1 + (30528*x0)), tmp15, rmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h7/ch7wocp4merqsyzxqegitglsk5s3i6tl3sxtaz5cfudwrhgrpxom.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_969, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_969, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
triton_red_fused_nll_loss_forward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr4 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.full([XBLOCK, RBLOCK], 30522, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 30522)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp8 < 30522")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (30528*r1) + (3907584*((r1 % 128) // 128)) + (250085376*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (tmp8), rmask & xmask, eviction_policy='evict_last')
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 - tmp13
        tmp16 = tl_math.log(tmp15)
        tmp17 = tmp14 - tmp16
        tmp18 = -tmp17
        tmp19 = 0.0
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp24 = tmp2.to(tl.int64)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp22, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp26, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/el/celbcz6vq6auvssxm5xvhbu6njj3lyyyhnriqeie3l7zkurcq5cv.py
# Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   masked_lm_loss => convert_element_type, div_48, full_default_3, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_969, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_969, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_48 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type), kwargs = {})
triton_per_fused_nll_loss_forward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_nll_loss_forward_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 128), (128, 1))
    assert_size_stride(arg1_1, (30522, 128), (128, 1))
    assert_size_stride(arg2_1, (512, 512), (512, 1))
    assert_size_stride(arg3_1, (2, 512), (512, 1))
    assert_size_stride(arg4_1, (512, 384), (384, 1))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (128, 128), (128, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, 128), (128, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, 512), (512, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, 128), (128, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (512, 128), (128, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (128, 512), (512, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (512, 128), (128, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (128, 512), (512, 1))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, 512), (512, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (512, 128), (128, 1))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (128, 512), (512, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (512, 128), (128, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (128, 512), (512, 1))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, 128), (128, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (128, 512), (512, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, 128), (128, 1))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, 128), (128, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, 512), (512, 1))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, 128), (128, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (512, 128), (128, 1))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (128, 512), (512, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (512, 128), (128, 1))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (128, 512), (512, 1))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, 512), (512, 1))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (512, 128), (128, 1))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (128, 512), (512, 1))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (512, 128), (128, 1))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (128, 512), (512, 1))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (512, 128), (128, 1))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (128, 512), (512, 1))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, 128), (128, 1))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, 128), (128, 1))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, 512), (512, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, 128), (128, 1))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (512, 128), (128, 1))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (128, 512), (512, 1))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (512, 128), (128, 1))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (128, 512), (512, 1))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, 512), (512, 1))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (512, 128), (128, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (128, 512), (512, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (512, 128), (128, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (128, 512), (512, 1))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (512, 128), (128, 1))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (128, 512), (512, 1))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (128, ), (1, ))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (128, 128), (128, 1))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, 128), (128, 1))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg150_1, (128, 512), (512, 1))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (128, 128), (128, 1))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (512, 128), (128, 1))
    assert_size_stride(arg157_1, (512, ), (1, ))
    assert_size_stride(arg158_1, (128, 512), (512, 1))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (128, ), (1, ))
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (512, 128), (128, 1))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (128, 512), (512, 1))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, 512), (512, 1))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (512, 128), (128, 1))
    assert_size_stride(arg175_1, (512, ), (1, ))
    assert_size_stride(arg176_1, (128, 512), (512, 1))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (512, 128), (128, 1))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (128, 512), (512, 1))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (512, 128), (128, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (128, 512), (512, 1))
    assert_size_stride(arg189_1, (128, ), (1, ))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (128, ), (1, ))
    assert_size_stride(arg192_1, (128, 128), (128, 1))
    assert_size_stride(arg193_1, (128, ), (1, ))
    assert_size_stride(arg194_1, (128, 128), (128, 1))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (128, 512), (512, 1))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, 128), (128, 1))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (512, 128), (128, 1))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (128, 512), (512, 1))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (512, 128), (128, 1))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (128, 512), (512, 1))
    assert_size_stride(arg213_1, (128, ), (1, ))
    assert_size_stride(arg214_1, (128, ), (1, ))
    assert_size_stride(arg215_1, (128, ), (1, ))
    assert_size_stride(arg216_1, (128, 512), (512, 1))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (128, ), (1, ))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (512, 128), (128, 1))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (128, 512), (512, 1))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (128, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (512, 128), (128, 1))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (128, 512), (512, 1))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (128, ), (1, ))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (512, 128), (128, 1))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (128, 512), (512, 1))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, 128), (128, 1))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (128, 128), (128, 1))
    assert_size_stride(arg241_1, (128, ), (1, ))
    assert_size_stride(arg242_1, (128, 512), (512, 1))
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, 128), (128, 1))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (512, 128), (128, 1))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (128, 512), (512, 1))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (512, 128), (128, 1))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (128, 512), (512, 1))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, 512), (512, 1))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (512, 128), (128, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (128, 512), (512, 1))
    assert_size_stride(arg269_1, (128, ), (1, ))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (128, ), (1, ))
    assert_size_stride(arg272_1, (512, 128), (128, 1))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (128, 512), (512, 1))
    assert_size_stride(arg275_1, (128, ), (1, ))
    assert_size_stride(arg276_1, (128, ), (1, ))
    assert_size_stride(arg277_1, (128, ), (1, ))
    assert_size_stride(arg278_1, (512, 128), (128, 1))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (128, 512), (512, 1))
    assert_size_stride(arg281_1, (128, ), (1, ))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (128, ), (1, ))
    assert_size_stride(arg284_1, (128, 128), (128, 1))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, 128), (128, 1))
    assert_size_stride(arg287_1, (128, ), (1, ))
    assert_size_stride(arg288_1, (128, 512), (512, 1))
    assert_size_stride(arg289_1, (128, ), (1, ))
    assert_size_stride(arg290_1, (128, 128), (128, 1))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (128, ), (1, ))
    assert_size_stride(arg294_1, (512, 128), (128, 1))
    assert_size_stride(arg295_1, (512, ), (1, ))
    assert_size_stride(arg296_1, (128, 512), (512, 1))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (128, ), (1, ))
    assert_size_stride(arg300_1, (512, 128), (128, 1))
    assert_size_stride(arg301_1, (512, ), (1, ))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (128, 512), (512, 1))
    assert_size_stride(arg305_1, (128, ), (1, ))
    assert_size_stride(arg306_1, (128, ), (1, ))
    assert_size_stride(arg307_1, (128, ), (1, ))
    assert_size_stride(arg308_1, (128, 512), (512, 1))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, ), (1, ))
    assert_size_stride(arg312_1, (512, 128), (128, 1))
    assert_size_stride(arg313_1, (512, ), (1, ))
    assert_size_stride(arg314_1, (128, 512), (512, 1))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (128, ), (1, ))
    assert_size_stride(arg317_1, (128, ), (1, ))
    assert_size_stride(arg318_1, (512, 128), (128, 1))
    assert_size_stride(arg319_1, (512, ), (1, ))
    assert_size_stride(arg320_1, (128, 512), (512, 1))
    assert_size_stride(arg321_1, (128, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (128, ), (1, ))
    assert_size_stride(arg324_1, (512, 128), (128, 1))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (128, 512), (512, 1))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (128, ), (1, ))
    assert_size_stride(arg330_1, (128, 128), (128, 1))
    assert_size_stride(arg331_1, (128, ), (1, ))
    assert_size_stride(arg332_1, (128, 128), (128, 1))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, 512), (512, 1))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (128, 128), (128, 1))
    assert_size_stride(arg337_1, (128, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (512, 128), (128, 1))
    assert_size_stride(arg341_1, (512, ), (1, ))
    assert_size_stride(arg342_1, (128, 512), (512, 1))
    assert_size_stride(arg343_1, (128, ), (1, ))
    assert_size_stride(arg344_1, (128, ), (1, ))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (512, 128), (128, 1))
    assert_size_stride(arg347_1, (512, ), (1, ))
    assert_size_stride(arg348_1, (512, ), (1, ))
    assert_size_stride(arg349_1, (512, ), (1, ))
    assert_size_stride(arg350_1, (128, 512), (512, 1))
    assert_size_stride(arg351_1, (128, ), (1, ))
    assert_size_stride(arg352_1, (128, ), (1, ))
    assert_size_stride(arg353_1, (128, ), (1, ))
    assert_size_stride(arg354_1, (128, 512), (512, 1))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (512, 128), (128, 1))
    assert_size_stride(arg359_1, (512, ), (1, ))
    assert_size_stride(arg360_1, (128, 512), (512, 1))
    assert_size_stride(arg361_1, (128, ), (1, ))
    assert_size_stride(arg362_1, (128, ), (1, ))
    assert_size_stride(arg363_1, (128, ), (1, ))
    assert_size_stride(arg364_1, (512, 128), (128, 1))
    assert_size_stride(arg365_1, (512, ), (1, ))
    assert_size_stride(arg366_1, (128, 512), (512, 1))
    assert_size_stride(arg367_1, (128, ), (1, ))
    assert_size_stride(arg368_1, (128, ), (1, ))
    assert_size_stride(arg369_1, (128, ), (1, ))
    assert_size_stride(arg370_1, (512, 128), (128, 1))
    assert_size_stride(arg371_1, (512, ), (1, ))
    assert_size_stride(arg372_1, (128, 512), (512, 1))
    assert_size_stride(arg373_1, (128, ), (1, ))
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (128, 128), (128, 1))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, 128), (128, 1))
    assert_size_stride(arg379_1, (128, ), (1, ))
    assert_size_stride(arg380_1, (128, 512), (512, 1))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (128, 128), (128, 1))
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (128, ), (1, ))
    assert_size_stride(arg385_1, (128, ), (1, ))
    assert_size_stride(arg386_1, (512, 128), (128, 1))
    assert_size_stride(arg387_1, (512, ), (1, ))
    assert_size_stride(arg388_1, (128, 512), (512, 1))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (128, ), (1, ))
    assert_size_stride(arg391_1, (128, ), (1, ))
    assert_size_stride(arg392_1, (512, 128), (128, 1))
    assert_size_stride(arg393_1, (512, ), (1, ))
    assert_size_stride(arg394_1, (512, ), (1, ))
    assert_size_stride(arg395_1, (512, ), (1, ))
    assert_size_stride(arg396_1, (128, 512), (512, 1))
    assert_size_stride(arg397_1, (128, ), (1, ))
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (128, 512), (512, 1))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (128, ), (1, ))
    assert_size_stride(arg403_1, (128, ), (1, ))
    assert_size_stride(arg404_1, (512, 128), (128, 1))
    assert_size_stride(arg405_1, (512, ), (1, ))
    assert_size_stride(arg406_1, (128, 512), (512, 1))
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (128, ), (1, ))
    assert_size_stride(arg409_1, (128, ), (1, ))
    assert_size_stride(arg410_1, (512, 128), (128, 1))
    assert_size_stride(arg411_1, (512, ), (1, ))
    assert_size_stride(arg412_1, (128, 512), (512, 1))
    assert_size_stride(arg413_1, (128, ), (1, ))
    assert_size_stride(arg414_1, (128, ), (1, ))
    assert_size_stride(arg415_1, (128, ), (1, ))
    assert_size_stride(arg416_1, (512, 128), (128, 1))
    assert_size_stride(arg417_1, (512, ), (1, ))
    assert_size_stride(arg418_1, (128, 512), (512, 1))
    assert_size_stride(arg419_1, (128, ), (1, ))
    assert_size_stride(arg420_1, (128, ), (1, ))
    assert_size_stride(arg421_1, (128, ), (1, ))
    assert_size_stride(arg422_1, (128, 128), (128, 1))
    assert_size_stride(arg423_1, (128, ), (1, ))
    assert_size_stride(arg424_1, (128, 128), (128, 1))
    assert_size_stride(arg425_1, (128, ), (1, ))
    assert_size_stride(arg426_1, (128, 512), (512, 1))
    assert_size_stride(arg427_1, (128, ), (1, ))
    assert_size_stride(arg428_1, (128, 128), (128, 1))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (128, ), (1, ))
    assert_size_stride(arg431_1, (128, ), (1, ))
    assert_size_stride(arg432_1, (512, 128), (128, 1))
    assert_size_stride(arg433_1, (512, ), (1, ))
    assert_size_stride(arg434_1, (128, 512), (512, 1))
    assert_size_stride(arg435_1, (128, ), (1, ))
    assert_size_stride(arg436_1, (128, ), (1, ))
    assert_size_stride(arg437_1, (128, ), (1, ))
    assert_size_stride(arg438_1, (512, 128), (128, 1))
    assert_size_stride(arg439_1, (512, ), (1, ))
    assert_size_stride(arg440_1, (512, ), (1, ))
    assert_size_stride(arg441_1, (512, ), (1, ))
    assert_size_stride(arg442_1, (128, 512), (512, 1))
    assert_size_stride(arg443_1, (128, ), (1, ))
    assert_size_stride(arg444_1, (128, ), (1, ))
    assert_size_stride(arg445_1, (128, ), (1, ))
    assert_size_stride(arg446_1, (128, 512), (512, 1))
    assert_size_stride(arg447_1, (128, ), (1, ))
    assert_size_stride(arg448_1, (128, ), (1, ))
    assert_size_stride(arg449_1, (128, ), (1, ))
    assert_size_stride(arg450_1, (512, 128), (128, 1))
    assert_size_stride(arg451_1, (512, ), (1, ))
    assert_size_stride(arg452_1, (128, 512), (512, 1))
    assert_size_stride(arg453_1, (128, ), (1, ))
    assert_size_stride(arg454_1, (128, ), (1, ))
    assert_size_stride(arg455_1, (128, ), (1, ))
    assert_size_stride(arg456_1, (512, 128), (128, 1))
    assert_size_stride(arg457_1, (512, ), (1, ))
    assert_size_stride(arg458_1, (128, 512), (512, 1))
    assert_size_stride(arg459_1, (128, ), (1, ))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (128, ), (1, ))
    assert_size_stride(arg462_1, (512, 128), (128, 1))
    assert_size_stride(arg463_1, (512, ), (1, ))
    assert_size_stride(arg464_1, (128, 512), (512, 1))
    assert_size_stride(arg465_1, (128, ), (1, ))
    assert_size_stride(arg466_1, (128, ), (1, ))
    assert_size_stride(arg467_1, (128, ), (1, ))
    assert_size_stride(arg468_1, (128, 128), (128, 1))
    assert_size_stride(arg469_1, (128, ), (1, ))
    assert_size_stride(arg470_1, (128, 128), (128, 1))
    assert_size_stride(arg471_1, (128, ), (1, ))
    assert_size_stride(arg472_1, (128, 512), (512, 1))
    assert_size_stride(arg473_1, (128, ), (1, ))
    assert_size_stride(arg474_1, (128, 128), (128, 1))
    assert_size_stride(arg475_1, (128, ), (1, ))
    assert_size_stride(arg476_1, (128, ), (1, ))
    assert_size_stride(arg477_1, (128, ), (1, ))
    assert_size_stride(arg478_1, (512, 128), (128, 1))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (128, 512), (512, 1))
    assert_size_stride(arg481_1, (128, ), (1, ))
    assert_size_stride(arg482_1, (128, ), (1, ))
    assert_size_stride(arg483_1, (128, ), (1, ))
    assert_size_stride(arg484_1, (512, 128), (128, 1))
    assert_size_stride(arg485_1, (512, ), (1, ))
    assert_size_stride(arg486_1, (512, ), (1, ))
    assert_size_stride(arg487_1, (512, ), (1, ))
    assert_size_stride(arg488_1, (128, 512), (512, 1))
    assert_size_stride(arg489_1, (128, ), (1, ))
    assert_size_stride(arg490_1, (128, ), (1, ))
    assert_size_stride(arg491_1, (128, ), (1, ))
    assert_size_stride(arg492_1, (128, 512), (512, 1))
    assert_size_stride(arg493_1, (128, ), (1, ))
    assert_size_stride(arg494_1, (128, ), (1, ))
    assert_size_stride(arg495_1, (128, ), (1, ))
    assert_size_stride(arg496_1, (512, 128), (128, 1))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (128, 512), (512, 1))
    assert_size_stride(arg499_1, (128, ), (1, ))
    assert_size_stride(arg500_1, (128, ), (1, ))
    assert_size_stride(arg501_1, (128, ), (1, ))
    assert_size_stride(arg502_1, (512, 128), (128, 1))
    assert_size_stride(arg503_1, (512, ), (1, ))
    assert_size_stride(arg504_1, (128, 512), (512, 1))
    assert_size_stride(arg505_1, (128, ), (1, ))
    assert_size_stride(arg506_1, (128, ), (1, ))
    assert_size_stride(arg507_1, (128, ), (1, ))
    assert_size_stride(arg508_1, (512, 128), (128, 1))
    assert_size_stride(arg509_1, (512, ), (1, ))
    assert_size_stride(arg510_1, (128, 512), (512, 1))
    assert_size_stride(arg511_1, (128, ), (1, ))
    assert_size_stride(arg512_1, (128, ), (1, ))
    assert_size_stride(arg513_1, (128, ), (1, ))
    assert_size_stride(arg514_1, (128, 128), (128, 1))
    assert_size_stride(arg515_1, (128, ), (1, ))
    assert_size_stride(arg516_1, (128, 128), (128, 1))
    assert_size_stride(arg517_1, (128, ), (1, ))
    assert_size_stride(arg518_1, (128, 512), (512, 1))
    assert_size_stride(arg519_1, (128, ), (1, ))
    assert_size_stride(arg520_1, (128, 128), (128, 1))
    assert_size_stride(arg521_1, (128, ), (1, ))
    assert_size_stride(arg522_1, (128, ), (1, ))
    assert_size_stride(arg523_1, (128, ), (1, ))
    assert_size_stride(arg524_1, (512, 128), (128, 1))
    assert_size_stride(arg525_1, (512, ), (1, ))
    assert_size_stride(arg526_1, (128, 512), (512, 1))
    assert_size_stride(arg527_1, (128, ), (1, ))
    assert_size_stride(arg528_1, (128, ), (1, ))
    assert_size_stride(arg529_1, (128, ), (1, ))
    assert_size_stride(arg530_1, (512, 128), (128, 1))
    assert_size_stride(arg531_1, (512, ), (1, ))
    assert_size_stride(arg532_1, (512, ), (1, ))
    assert_size_stride(arg533_1, (512, ), (1, ))
    assert_size_stride(arg534_1, (128, 512), (512, 1))
    assert_size_stride(arg535_1, (128, ), (1, ))
    assert_size_stride(arg536_1, (128, ), (1, ))
    assert_size_stride(arg537_1, (128, ), (1, ))
    assert_size_stride(arg538_1, (128, 512), (512, 1))
    assert_size_stride(arg539_1, (128, ), (1, ))
    assert_size_stride(arg540_1, (128, ), (1, ))
    assert_size_stride(arg541_1, (128, ), (1, ))
    assert_size_stride(arg542_1, (512, 128), (128, 1))
    assert_size_stride(arg543_1, (512, ), (1, ))
    assert_size_stride(arg544_1, (128, 512), (512, 1))
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (128, ), (1, ))
    assert_size_stride(arg547_1, (128, ), (1, ))
    assert_size_stride(arg548_1, (512, 128), (128, 1))
    assert_size_stride(arg549_1, (512, ), (1, ))
    assert_size_stride(arg550_1, (128, 512), (512, 1))
    assert_size_stride(arg551_1, (128, ), (1, ))
    assert_size_stride(arg552_1, (128, ), (1, ))
    assert_size_stride(arg553_1, (128, ), (1, ))
    assert_size_stride(arg554_1, (512, 128), (128, 1))
    assert_size_stride(arg555_1, (512, ), (1, ))
    assert_size_stride(arg556_1, (128, 512), (512, 1))
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (128, ), (1, ))
    assert_size_stride(arg559_1, (128, ), (1, ))
    assert_size_stride(arg560_1, (128, 128), (128, 1))
    assert_size_stride(arg561_1, (128, ), (1, ))
    assert_size_stride(arg562_1, (128, 128), (128, 1))
    assert_size_stride(arg563_1, (128, ), (1, ))
    assert_size_stride(arg564_1, (128, 512), (512, 1))
    assert_size_stride(arg565_1, (128, ), (1, ))
    assert_size_stride(arg566_1, (128, 128), (128, 1))
    assert_size_stride(arg567_1, (128, ), (1, ))
    assert_size_stride(arg568_1, (128, ), (1, ))
    assert_size_stride(arg569_1, (128, ), (1, ))
    assert_size_stride(arg570_1, (512, 128), (128, 1))
    assert_size_stride(arg571_1, (512, ), (1, ))
    assert_size_stride(arg572_1, (128, 512), (512, 1))
    assert_size_stride(arg573_1, (128, ), (1, ))
    assert_size_stride(arg574_1, (128, ), (1, ))
    assert_size_stride(arg575_1, (128, ), (1, ))
    assert_size_stride(arg576_1, (512, 128), (128, 1))
    assert_size_stride(arg577_1, (512, ), (1, ))
    assert_size_stride(arg578_1, (512, ), (1, ))
    assert_size_stride(arg579_1, (512, ), (1, ))
    assert_size_stride(arg580_1, (128, 512), (512, 1))
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (128, ), (1, ))
    assert_size_stride(arg583_1, (128, ), (1, ))
    assert_size_stride(arg584_1, (128, 512), (512, 1))
    assert_size_stride(arg585_1, (128, ), (1, ))
    assert_size_stride(arg586_1, (128, ), (1, ))
    assert_size_stride(arg587_1, (128, ), (1, ))
    assert_size_stride(arg588_1, (512, 128), (128, 1))
    assert_size_stride(arg589_1, (512, ), (1, ))
    assert_size_stride(arg590_1, (128, 512), (512, 1))
    assert_size_stride(arg591_1, (128, ), (1, ))
    assert_size_stride(arg592_1, (128, ), (1, ))
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (512, 128), (128, 1))
    assert_size_stride(arg595_1, (512, ), (1, ))
    assert_size_stride(arg596_1, (128, 512), (512, 1))
    assert_size_stride(arg597_1, (128, ), (1, ))
    assert_size_stride(arg598_1, (128, ), (1, ))
    assert_size_stride(arg599_1, (128, ), (1, ))
    assert_size_stride(arg600_1, (512, 128), (128, 1))
    assert_size_stride(arg601_1, (512, ), (1, ))
    assert_size_stride(arg602_1, (128, 512), (512, 1))
    assert_size_stride(arg603_1, (128, ), (1, ))
    assert_size_stride(arg604_1, (128, ), (1, ))
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (128, 128), (128, 1))
    assert_size_stride(arg607_1, (128, ), (1, ))
    assert_size_stride(arg608_1, (128, 128), (128, 1))
    assert_size_stride(arg609_1, (128, ), (1, ))
    assert_size_stride(arg610_1, (128, 512), (512, 1))
    assert_size_stride(arg611_1, (128, ), (1, ))
    assert_size_stride(arg612_1, (128, 128), (128, 1))
    assert_size_stride(arg613_1, (128, ), (1, ))
    assert_size_stride(arg614_1, (128, ), (1, ))
    assert_size_stride(arg615_1, (128, ), (1, ))
    assert_size_stride(arg616_1, (512, 128), (128, 1))
    assert_size_stride(arg617_1, (512, ), (1, ))
    assert_size_stride(arg618_1, (128, 512), (512, 1))
    assert_size_stride(arg619_1, (128, ), (1, ))
    assert_size_stride(arg620_1, (128, ), (1, ))
    assert_size_stride(arg621_1, (128, ), (1, ))
    assert_size_stride(arg622_1, (512, 128), (128, 1))
    assert_size_stride(arg623_1, (512, ), (1, ))
    assert_size_stride(arg624_1, (512, ), (1, ))
    assert_size_stride(arg625_1, (512, ), (1, ))
    assert_size_stride(arg626_1, (128, 512), (512, 1))
    assert_size_stride(arg627_1, (128, ), (1, ))
    assert_size_stride(arg628_1, (128, ), (1, ))
    assert_size_stride(arg629_1, (128, ), (1, ))
    assert_size_stride(arg630_1, (128, 512), (512, 1))
    assert_size_stride(arg631_1, (128, ), (1, ))
    assert_size_stride(arg632_1, (128, ), (1, ))
    assert_size_stride(arg633_1, (128, ), (1, ))
    assert_size_stride(arg634_1, (512, 128), (128, 1))
    assert_size_stride(arg635_1, (512, ), (1, ))
    assert_size_stride(arg636_1, (128, 512), (512, 1))
    assert_size_stride(arg637_1, (128, ), (1, ))
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (128, ), (1, ))
    assert_size_stride(arg640_1, (512, 128), (128, 1))
    assert_size_stride(arg641_1, (512, ), (1, ))
    assert_size_stride(arg642_1, (128, 512), (512, 1))
    assert_size_stride(arg643_1, (128, ), (1, ))
    assert_size_stride(arg644_1, (128, ), (1, ))
    assert_size_stride(arg645_1, (128, ), (1, ))
    assert_size_stride(arg646_1, (512, 128), (128, 1))
    assert_size_stride(arg647_1, (512, ), (1, ))
    assert_size_stride(arg648_1, (128, 512), (512, 1))
    assert_size_stride(arg649_1, (128, ), (1, ))
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (128, ), (1, ))
    assert_size_stride(arg652_1, (128, 128), (128, 1))
    assert_size_stride(arg653_1, (128, ), (1, ))
    assert_size_stride(arg654_1, (128, 128), (128, 1))
    assert_size_stride(arg655_1, (128, ), (1, ))
    assert_size_stride(arg656_1, (128, 512), (512, 1))
    assert_size_stride(arg657_1, (128, ), (1, ))
    assert_size_stride(arg658_1, (128, 128), (128, 1))
    assert_size_stride(arg659_1, (128, ), (1, ))
    assert_size_stride(arg660_1, (128, ), (1, ))
    assert_size_stride(arg661_1, (128, ), (1, ))
    assert_size_stride(arg662_1, (512, 128), (128, 1))
    assert_size_stride(arg663_1, (512, ), (1, ))
    assert_size_stride(arg664_1, (128, 512), (512, 1))
    assert_size_stride(arg665_1, (128, ), (1, ))
    assert_size_stride(arg666_1, (128, ), (1, ))
    assert_size_stride(arg667_1, (128, ), (1, ))
    assert_size_stride(arg668_1, (512, 128), (128, 1))
    assert_size_stride(arg669_1, (512, ), (1, ))
    assert_size_stride(arg670_1, (512, ), (1, ))
    assert_size_stride(arg671_1, (512, ), (1, ))
    assert_size_stride(arg672_1, (128, 512), (512, 1))
    assert_size_stride(arg673_1, (128, ), (1, ))
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (128, ), (1, ))
    assert_size_stride(arg676_1, (128, 512), (512, 1))
    assert_size_stride(arg677_1, (128, ), (1, ))
    assert_size_stride(arg678_1, (128, ), (1, ))
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (512, 128), (128, 1))
    assert_size_stride(arg681_1, (512, ), (1, ))
    assert_size_stride(arg682_1, (128, 512), (512, 1))
    assert_size_stride(arg683_1, (128, ), (1, ))
    assert_size_stride(arg684_1, (128, ), (1, ))
    assert_size_stride(arg685_1, (128, ), (1, ))
    assert_size_stride(arg686_1, (512, 128), (128, 1))
    assert_size_stride(arg687_1, (512, ), (1, ))
    assert_size_stride(arg688_1, (128, 512), (512, 1))
    assert_size_stride(arg689_1, (128, ), (1, ))
    assert_size_stride(arg690_1, (128, ), (1, ))
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (512, 128), (128, 1))
    assert_size_stride(arg693_1, (512, ), (1, ))
    assert_size_stride(arg694_1, (128, 512), (512, 1))
    assert_size_stride(arg695_1, (128, ), (1, ))
    assert_size_stride(arg696_1, (128, ), (1, ))
    assert_size_stride(arg697_1, (128, ), (1, ))
    assert_size_stride(arg698_1, (128, 128), (128, 1))
    assert_size_stride(arg699_1, (128, ), (1, ))
    assert_size_stride(arg700_1, (128, 128), (128, 1))
    assert_size_stride(arg701_1, (128, ), (1, ))
    assert_size_stride(arg702_1, (128, 512), (512, 1))
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (128, 128), (128, 1))
    assert_size_stride(arg705_1, (128, ), (1, ))
    assert_size_stride(arg706_1, (128, ), (1, ))
    assert_size_stride(arg707_1, (128, ), (1, ))
    assert_size_stride(arg708_1, (512, 128), (128, 1))
    assert_size_stride(arg709_1, (512, ), (1, ))
    assert_size_stride(arg710_1, (128, 512), (512, 1))
    assert_size_stride(arg711_1, (128, ), (1, ))
    assert_size_stride(arg712_1, (128, ), (1, ))
    assert_size_stride(arg713_1, (128, ), (1, ))
    assert_size_stride(arg714_1, (512, 128), (128, 1))
    assert_size_stride(arg715_1, (512, ), (1, ))
    assert_size_stride(arg716_1, (512, ), (1, ))
    assert_size_stride(arg717_1, (512, ), (1, ))
    assert_size_stride(arg718_1, (128, 512), (512, 1))
    assert_size_stride(arg719_1, (128, ), (1, ))
    assert_size_stride(arg720_1, (128, ), (1, ))
    assert_size_stride(arg721_1, (128, ), (1, ))
    assert_size_stride(arg722_1, (128, 512), (512, 1))
    assert_size_stride(arg723_1, (128, ), (1, ))
    assert_size_stride(arg724_1, (128, ), (1, ))
    assert_size_stride(arg725_1, (128, ), (1, ))
    assert_size_stride(arg726_1, (512, 128), (128, 1))
    assert_size_stride(arg727_1, (512, ), (1, ))
    assert_size_stride(arg728_1, (128, 512), (512, 1))
    assert_size_stride(arg729_1, (128, ), (1, ))
    assert_size_stride(arg730_1, (128, ), (1, ))
    assert_size_stride(arg731_1, (128, ), (1, ))
    assert_size_stride(arg732_1, (512, 128), (128, 1))
    assert_size_stride(arg733_1, (512, ), (1, ))
    assert_size_stride(arg734_1, (128, 512), (512, 1))
    assert_size_stride(arg735_1, (128, ), (1, ))
    assert_size_stride(arg736_1, (128, ), (1, ))
    assert_size_stride(arg737_1, (128, ), (1, ))
    assert_size_stride(arg738_1, (512, 128), (128, 1))
    assert_size_stride(arg739_1, (512, ), (1, ))
    assert_size_stride(arg740_1, (128, 512), (512, 1))
    assert_size_stride(arg741_1, (128, ), (1, ))
    assert_size_stride(arg742_1, (128, ), (1, ))
    assert_size_stride(arg743_1, (128, ), (1, ))
    assert_size_stride(arg744_1, (128, 128), (128, 1))
    assert_size_stride(arg745_1, (128, ), (1, ))
    assert_size_stride(arg746_1, (128, 128), (128, 1))
    assert_size_stride(arg747_1, (128, ), (1, ))
    assert_size_stride(arg748_1, (128, 512), (512, 1))
    assert_size_stride(arg749_1, (128, ), (1, ))
    assert_size_stride(arg750_1, (128, 128), (128, 1))
    assert_size_stride(arg751_1, (128, ), (1, ))
    assert_size_stride(arg752_1, (128, ), (1, ))
    assert_size_stride(arg753_1, (128, ), (1, ))
    assert_size_stride(arg754_1, (512, 128), (128, 1))
    assert_size_stride(arg755_1, (512, ), (1, ))
    assert_size_stride(arg756_1, (128, 512), (512, 1))
    assert_size_stride(arg757_1, (128, ), (1, ))
    assert_size_stride(arg758_1, (128, ), (1, ))
    assert_size_stride(arg759_1, (128, ), (1, ))
    assert_size_stride(arg760_1, (512, 128), (128, 1))
    assert_size_stride(arg761_1, (512, ), (1, ))
    assert_size_stride(arg762_1, (512, ), (1, ))
    assert_size_stride(arg763_1, (512, ), (1, ))
    assert_size_stride(arg764_1, (128, 512), (512, 1))
    assert_size_stride(arg765_1, (128, ), (1, ))
    assert_size_stride(arg766_1, (128, ), (1, ))
    assert_size_stride(arg767_1, (128, ), (1, ))
    assert_size_stride(arg768_1, (128, 512), (512, 1))
    assert_size_stride(arg769_1, (128, ), (1, ))
    assert_size_stride(arg770_1, (128, ), (1, ))
    assert_size_stride(arg771_1, (128, ), (1, ))
    assert_size_stride(arg772_1, (512, 128), (128, 1))
    assert_size_stride(arg773_1, (512, ), (1, ))
    assert_size_stride(arg774_1, (128, 512), (512, 1))
    assert_size_stride(arg775_1, (128, ), (1, ))
    assert_size_stride(arg776_1, (128, ), (1, ))
    assert_size_stride(arg777_1, (128, ), (1, ))
    assert_size_stride(arg778_1, (512, 128), (128, 1))
    assert_size_stride(arg779_1, (512, ), (1, ))
    assert_size_stride(arg780_1, (128, 512), (512, 1))
    assert_size_stride(arg781_1, (128, ), (1, ))
    assert_size_stride(arg782_1, (128, ), (1, ))
    assert_size_stride(arg783_1, (128, ), (1, ))
    assert_size_stride(arg784_1, (512, 128), (128, 1))
    assert_size_stride(arg785_1, (512, ), (1, ))
    assert_size_stride(arg786_1, (128, 512), (512, 1))
    assert_size_stride(arg787_1, (128, ), (1, ))
    assert_size_stride(arg788_1, (128, ), (1, ))
    assert_size_stride(arg789_1, (128, ), (1, ))
    assert_size_stride(arg790_1, (128, 128), (128, 1))
    assert_size_stride(arg791_1, (128, ), (1, ))
    assert_size_stride(arg792_1, (128, 128), (128, 1))
    assert_size_stride(arg793_1, (128, ), (1, ))
    assert_size_stride(arg794_1, (128, 512), (512, 1))
    assert_size_stride(arg795_1, (128, ), (1, ))
    assert_size_stride(arg796_1, (128, 128), (128, 1))
    assert_size_stride(arg797_1, (128, ), (1, ))
    assert_size_stride(arg798_1, (128, ), (1, ))
    assert_size_stride(arg799_1, (128, ), (1, ))
    assert_size_stride(arg800_1, (512, 128), (128, 1))
    assert_size_stride(arg801_1, (512, ), (1, ))
    assert_size_stride(arg802_1, (128, 512), (512, 1))
    assert_size_stride(arg803_1, (128, ), (1, ))
    assert_size_stride(arg804_1, (128, ), (1, ))
    assert_size_stride(arg805_1, (128, ), (1, ))
    assert_size_stride(arg806_1, (512, 128), (128, 1))
    assert_size_stride(arg807_1, (512, ), (1, ))
    assert_size_stride(arg808_1, (512, ), (1, ))
    assert_size_stride(arg809_1, (512, ), (1, ))
    assert_size_stride(arg810_1, (128, 512), (512, 1))
    assert_size_stride(arg811_1, (128, ), (1, ))
    assert_size_stride(arg812_1, (128, ), (1, ))
    assert_size_stride(arg813_1, (128, ), (1, ))
    assert_size_stride(arg814_1, (128, 512), (512, 1))
    assert_size_stride(arg815_1, (128, ), (1, ))
    assert_size_stride(arg816_1, (128, ), (1, ))
    assert_size_stride(arg817_1, (128, ), (1, ))
    assert_size_stride(arg818_1, (512, 128), (128, 1))
    assert_size_stride(arg819_1, (512, ), (1, ))
    assert_size_stride(arg820_1, (128, 512), (512, 1))
    assert_size_stride(arg821_1, (128, ), (1, ))
    assert_size_stride(arg822_1, (128, ), (1, ))
    assert_size_stride(arg823_1, (128, ), (1, ))
    assert_size_stride(arg824_1, (512, 128), (128, 1))
    assert_size_stride(arg825_1, (512, ), (1, ))
    assert_size_stride(arg826_1, (128, 512), (512, 1))
    assert_size_stride(arg827_1, (128, ), (1, ))
    assert_size_stride(arg828_1, (128, ), (1, ))
    assert_size_stride(arg829_1, (128, ), (1, ))
    assert_size_stride(arg830_1, (512, 128), (128, 1))
    assert_size_stride(arg831_1, (512, ), (1, ))
    assert_size_stride(arg832_1, (128, 512), (512, 1))
    assert_size_stride(arg833_1, (128, ), (1, ))
    assert_size_stride(arg834_1, (128, ), (1, ))
    assert_size_stride(arg835_1, (128, ), (1, ))
    assert_size_stride(arg836_1, (128, 128), (128, 1))
    assert_size_stride(arg837_1, (128, ), (1, ))
    assert_size_stride(arg838_1, (128, 128), (128, 1))
    assert_size_stride(arg839_1, (128, ), (1, ))
    assert_size_stride(arg840_1, (128, 512), (512, 1))
    assert_size_stride(arg841_1, (128, ), (1, ))
    assert_size_stride(arg842_1, (128, 128), (128, 1))
    assert_size_stride(arg843_1, (128, ), (1, ))
    assert_size_stride(arg844_1, (128, ), (1, ))
    assert_size_stride(arg845_1, (128, ), (1, ))
    assert_size_stride(arg846_1, (512, 128), (128, 1))
    assert_size_stride(arg847_1, (512, ), (1, ))
    assert_size_stride(arg848_1, (128, 512), (512, 1))
    assert_size_stride(arg849_1, (128, ), (1, ))
    assert_size_stride(arg850_1, (128, ), (1, ))
    assert_size_stride(arg851_1, (128, ), (1, ))
    assert_size_stride(arg852_1, (512, 128), (128, 1))
    assert_size_stride(arg853_1, (512, ), (1, ))
    assert_size_stride(arg854_1, (512, ), (1, ))
    assert_size_stride(arg855_1, (512, ), (1, ))
    assert_size_stride(arg856_1, (128, 512), (512, 1))
    assert_size_stride(arg857_1, (128, ), (1, ))
    assert_size_stride(arg858_1, (128, ), (1, ))
    assert_size_stride(arg859_1, (128, ), (1, ))
    assert_size_stride(arg860_1, (128, 512), (512, 1))
    assert_size_stride(arg861_1, (128, ), (1, ))
    assert_size_stride(arg862_1, (128, ), (1, ))
    assert_size_stride(arg863_1, (128, ), (1, ))
    assert_size_stride(arg864_1, (512, 128), (128, 1))
    assert_size_stride(arg865_1, (512, ), (1, ))
    assert_size_stride(arg866_1, (128, 512), (512, 1))
    assert_size_stride(arg867_1, (128, ), (1, ))
    assert_size_stride(arg868_1, (128, ), (1, ))
    assert_size_stride(arg869_1, (128, ), (1, ))
    assert_size_stride(arg870_1, (512, 128), (128, 1))
    assert_size_stride(arg871_1, (512, ), (1, ))
    assert_size_stride(arg872_1, (128, 512), (512, 1))
    assert_size_stride(arg873_1, (128, ), (1, ))
    assert_size_stride(arg874_1, (128, ), (1, ))
    assert_size_stride(arg875_1, (128, ), (1, ))
    assert_size_stride(arg876_1, (512, 128), (128, 1))
    assert_size_stride(arg877_1, (512, ), (1, ))
    assert_size_stride(arg878_1, (128, 512), (512, 1))
    assert_size_stride(arg879_1, (128, ), (1, ))
    assert_size_stride(arg880_1, (128, ), (1, ))
    assert_size_stride(arg881_1, (128, ), (1, ))
    assert_size_stride(arg882_1, (128, 128), (128, 1))
    assert_size_stride(arg883_1, (128, ), (1, ))
    assert_size_stride(arg884_1, (128, 128), (128, 1))
    assert_size_stride(arg885_1, (128, ), (1, ))
    assert_size_stride(arg886_1, (128, 512), (512, 1))
    assert_size_stride(arg887_1, (128, ), (1, ))
    assert_size_stride(arg888_1, (128, 128), (128, 1))
    assert_size_stride(arg889_1, (128, ), (1, ))
    assert_size_stride(arg890_1, (128, ), (1, ))
    assert_size_stride(arg891_1, (128, ), (1, ))
    assert_size_stride(arg892_1, (512, 128), (128, 1))
    assert_size_stride(arg893_1, (512, ), (1, ))
    assert_size_stride(arg894_1, (128, 512), (512, 1))
    assert_size_stride(arg895_1, (128, ), (1, ))
    assert_size_stride(arg896_1, (128, ), (1, ))
    assert_size_stride(arg897_1, (128, ), (1, ))
    assert_size_stride(arg898_1, (512, 128), (128, 1))
    assert_size_stride(arg899_1, (512, ), (1, ))
    assert_size_stride(arg900_1, (512, ), (1, ))
    assert_size_stride(arg901_1, (512, ), (1, ))
    assert_size_stride(arg902_1, (128, 512), (512, 1))
    assert_size_stride(arg903_1, (128, ), (1, ))
    assert_size_stride(arg904_1, (128, ), (1, ))
    assert_size_stride(arg905_1, (128, ), (1, ))
    assert_size_stride(arg906_1, (128, 512), (512, 1))
    assert_size_stride(arg907_1, (128, ), (1, ))
    assert_size_stride(arg908_1, (128, ), (1, ))
    assert_size_stride(arg909_1, (128, ), (1, ))
    assert_size_stride(arg910_1, (512, 128), (128, 1))
    assert_size_stride(arg911_1, (512, ), (1, ))
    assert_size_stride(arg912_1, (128, 512), (512, 1))
    assert_size_stride(arg913_1, (128, ), (1, ))
    assert_size_stride(arg914_1, (128, ), (1, ))
    assert_size_stride(arg915_1, (128, ), (1, ))
    assert_size_stride(arg916_1, (512, 128), (128, 1))
    assert_size_stride(arg917_1, (512, ), (1, ))
    assert_size_stride(arg918_1, (128, 512), (512, 1))
    assert_size_stride(arg919_1, (128, ), (1, ))
    assert_size_stride(arg920_1, (128, ), (1, ))
    assert_size_stride(arg921_1, (128, ), (1, ))
    assert_size_stride(arg922_1, (512, 128), (128, 1))
    assert_size_stride(arg923_1, (512, ), (1, ))
    assert_size_stride(arg924_1, (128, 512), (512, 1))
    assert_size_stride(arg925_1, (128, ), (1, ))
    assert_size_stride(arg926_1, (128, ), (1, ))
    assert_size_stride(arg927_1, (128, ), (1, ))
    assert_size_stride(arg928_1, (128, 128), (128, 1))
    assert_size_stride(arg929_1, (128, ), (1, ))
    assert_size_stride(arg930_1, (128, 128), (128, 1))
    assert_size_stride(arg931_1, (128, ), (1, ))
    assert_size_stride(arg932_1, (128, 512), (512, 1))
    assert_size_stride(arg933_1, (128, ), (1, ))
    assert_size_stride(arg934_1, (128, 128), (128, 1))
    assert_size_stride(arg935_1, (128, ), (1, ))
    assert_size_stride(arg936_1, (128, ), (1, ))
    assert_size_stride(arg937_1, (128, ), (1, ))
    assert_size_stride(arg938_1, (512, 128), (128, 1))
    assert_size_stride(arg939_1, (512, ), (1, ))
    assert_size_stride(arg940_1, (128, 512), (512, 1))
    assert_size_stride(arg941_1, (128, ), (1, ))
    assert_size_stride(arg942_1, (128, ), (1, ))
    assert_size_stride(arg943_1, (128, ), (1, ))
    assert_size_stride(arg944_1, (512, 128), (128, 1))
    assert_size_stride(arg945_1, (512, ), (1, ))
    assert_size_stride(arg946_1, (512, ), (1, ))
    assert_size_stride(arg947_1, (512, ), (1, ))
    assert_size_stride(arg948_1, (128, 512), (512, 1))
    assert_size_stride(arg949_1, (128, ), (1, ))
    assert_size_stride(arg950_1, (128, ), (1, ))
    assert_size_stride(arg951_1, (128, ), (1, ))
    assert_size_stride(arg952_1, (128, 512), (512, 1))
    assert_size_stride(arg953_1, (128, ), (1, ))
    assert_size_stride(arg954_1, (128, ), (1, ))
    assert_size_stride(arg955_1, (128, ), (1, ))
    assert_size_stride(arg956_1, (512, 128), (128, 1))
    assert_size_stride(arg957_1, (512, ), (1, ))
    assert_size_stride(arg958_1, (128, 512), (512, 1))
    assert_size_stride(arg959_1, (128, ), (1, ))
    assert_size_stride(arg960_1, (128, ), (1, ))
    assert_size_stride(arg961_1, (128, ), (1, ))
    assert_size_stride(arg962_1, (512, 128), (128, 1))
    assert_size_stride(arg963_1, (512, ), (1, ))
    assert_size_stride(arg964_1, (128, 512), (512, 1))
    assert_size_stride(arg965_1, (128, ), (1, ))
    assert_size_stride(arg966_1, (128, ), (1, ))
    assert_size_stride(arg967_1, (128, ), (1, ))
    assert_size_stride(arg968_1, (512, 128), (128, 1))
    assert_size_stride(arg969_1, (512, ), (1, ))
    assert_size_stride(arg970_1, (128, 512), (512, 1))
    assert_size_stride(arg971_1, (128, ), (1, ))
    assert_size_stride(arg972_1, (128, ), (1, ))
    assert_size_stride(arg973_1, (128, ), (1, ))
    assert_size_stride(arg974_1, (128, 128), (128, 1))
    assert_size_stride(arg975_1, (128, ), (1, ))
    assert_size_stride(arg976_1, (128, 128), (128, 1))
    assert_size_stride(arg977_1, (128, ), (1, ))
    assert_size_stride(arg978_1, (128, 512), (512, 1))
    assert_size_stride(arg979_1, (128, ), (1, ))
    assert_size_stride(arg980_1, (128, 128), (128, 1))
    assert_size_stride(arg981_1, (128, ), (1, ))
    assert_size_stride(arg982_1, (128, ), (1, ))
    assert_size_stride(arg983_1, (128, ), (1, ))
    assert_size_stride(arg984_1, (512, 128), (128, 1))
    assert_size_stride(arg985_1, (512, ), (1, ))
    assert_size_stride(arg986_1, (128, 512), (512, 1))
    assert_size_stride(arg987_1, (128, ), (1, ))
    assert_size_stride(arg988_1, (128, ), (1, ))
    assert_size_stride(arg989_1, (128, ), (1, ))
    assert_size_stride(arg990_1, (512, 128), (128, 1))
    assert_size_stride(arg991_1, (512, ), (1, ))
    assert_size_stride(arg992_1, (512, ), (1, ))
    assert_size_stride(arg993_1, (512, ), (1, ))
    assert_size_stride(arg994_1, (128, 512), (512, 1))
    assert_size_stride(arg995_1, (128, ), (1, ))
    assert_size_stride(arg996_1, (128, ), (1, ))
    assert_size_stride(arg997_1, (128, ), (1, ))
    assert_size_stride(arg998_1, (128, 512), (512, 1))
    assert_size_stride(arg999_1, (128, ), (1, ))
    assert_size_stride(arg1000_1, (128, ), (1, ))
    assert_size_stride(arg1001_1, (128, ), (1, ))
    assert_size_stride(arg1002_1, (512, 128), (128, 1))
    assert_size_stride(arg1003_1, (512, ), (1, ))
    assert_size_stride(arg1004_1, (128, 512), (512, 1))
    assert_size_stride(arg1005_1, (128, ), (1, ))
    assert_size_stride(arg1006_1, (128, ), (1, ))
    assert_size_stride(arg1007_1, (128, ), (1, ))
    assert_size_stride(arg1008_1, (512, 128), (128, 1))
    assert_size_stride(arg1009_1, (512, ), (1, ))
    assert_size_stride(arg1010_1, (128, 512), (512, 1))
    assert_size_stride(arg1011_1, (128, ), (1, ))
    assert_size_stride(arg1012_1, (128, ), (1, ))
    assert_size_stride(arg1013_1, (128, ), (1, ))
    assert_size_stride(arg1014_1, (512, 128), (128, 1))
    assert_size_stride(arg1015_1, (512, ), (1, ))
    assert_size_stride(arg1016_1, (128, 512), (512, 1))
    assert_size_stride(arg1017_1, (128, ), (1, ))
    assert_size_stride(arg1018_1, (128, ), (1, ))
    assert_size_stride(arg1019_1, (128, ), (1, ))
    assert_size_stride(arg1020_1, (128, 128), (128, 1))
    assert_size_stride(arg1021_1, (128, ), (1, ))
    assert_size_stride(arg1022_1, (128, 128), (128, 1))
    assert_size_stride(arg1023_1, (128, ), (1, ))
    assert_size_stride(arg1024_1, (128, 512), (512, 1))
    assert_size_stride(arg1025_1, (128, ), (1, ))
    assert_size_stride(arg1026_1, (128, 128), (128, 1))
    assert_size_stride(arg1027_1, (128, ), (1, ))
    assert_size_stride(arg1028_1, (128, ), (1, ))
    assert_size_stride(arg1029_1, (128, ), (1, ))
    assert_size_stride(arg1030_1, (512, 128), (128, 1))
    assert_size_stride(arg1031_1, (512, ), (1, ))
    assert_size_stride(arg1032_1, (128, 512), (512, 1))
    assert_size_stride(arg1033_1, (128, ), (1, ))
    assert_size_stride(arg1034_1, (128, ), (1, ))
    assert_size_stride(arg1035_1, (128, ), (1, ))
    assert_size_stride(arg1036_1, (512, 128), (128, 1))
    assert_size_stride(arg1037_1, (512, ), (1, ))
    assert_size_stride(arg1038_1, (512, ), (1, ))
    assert_size_stride(arg1039_1, (512, ), (1, ))
    assert_size_stride(arg1040_1, (128, 512), (512, 1))
    assert_size_stride(arg1041_1, (128, ), (1, ))
    assert_size_stride(arg1042_1, (128, ), (1, ))
    assert_size_stride(arg1043_1, (128, ), (1, ))
    assert_size_stride(arg1044_1, (128, 512), (512, 1))
    assert_size_stride(arg1045_1, (128, ), (1, ))
    assert_size_stride(arg1046_1, (128, ), (1, ))
    assert_size_stride(arg1047_1, (128, ), (1, ))
    assert_size_stride(arg1048_1, (512, 128), (128, 1))
    assert_size_stride(arg1049_1, (512, ), (1, ))
    assert_size_stride(arg1050_1, (128, 512), (512, 1))
    assert_size_stride(arg1051_1, (128, ), (1, ))
    assert_size_stride(arg1052_1, (128, ), (1, ))
    assert_size_stride(arg1053_1, (128, ), (1, ))
    assert_size_stride(arg1054_1, (512, 128), (128, 1))
    assert_size_stride(arg1055_1, (512, ), (1, ))
    assert_size_stride(arg1056_1, (128, 512), (512, 1))
    assert_size_stride(arg1057_1, (128, ), (1, ))
    assert_size_stride(arg1058_1, (128, ), (1, ))
    assert_size_stride(arg1059_1, (128, ), (1, ))
    assert_size_stride(arg1060_1, (512, 128), (128, 1))
    assert_size_stride(arg1061_1, (512, ), (1, ))
    assert_size_stride(arg1062_1, (128, 512), (512, 1))
    assert_size_stride(arg1063_1, (128, ), (1, ))
    assert_size_stride(arg1064_1, (128, ), (1, ))
    assert_size_stride(arg1065_1, (128, ), (1, ))
    assert_size_stride(arg1066_1, (128, 128), (128, 1))
    assert_size_stride(arg1067_1, (128, ), (1, ))
    assert_size_stride(arg1068_1, (128, 128), (128, 1))
    assert_size_stride(arg1069_1, (128, ), (1, ))
    assert_size_stride(arg1070_1, (128, 512), (512, 1))
    assert_size_stride(arg1071_1, (128, ), (1, ))
    assert_size_stride(arg1072_1, (128, 128), (128, 1))
    assert_size_stride(arg1073_1, (128, ), (1, ))
    assert_size_stride(arg1074_1, (128, ), (1, ))
    assert_size_stride(arg1075_1, (128, ), (1, ))
    assert_size_stride(arg1076_1, (512, 128), (128, 1))
    assert_size_stride(arg1077_1, (512, ), (1, ))
    assert_size_stride(arg1078_1, (128, 512), (512, 1))
    assert_size_stride(arg1079_1, (128, ), (1, ))
    assert_size_stride(arg1080_1, (128, ), (1, ))
    assert_size_stride(arg1081_1, (128, ), (1, ))
    assert_size_stride(arg1082_1, (512, 128), (128, 1))
    assert_size_stride(arg1083_1, (512, ), (1, ))
    assert_size_stride(arg1084_1, (512, ), (1, ))
    assert_size_stride(arg1085_1, (512, ), (1, ))
    assert_size_stride(arg1086_1, (128, 512), (512, 1))
    assert_size_stride(arg1087_1, (128, ), (1, ))
    assert_size_stride(arg1088_1, (128, ), (1, ))
    assert_size_stride(arg1089_1, (128, ), (1, ))
    assert_size_stride(arg1090_1, (128, 512), (512, 1))
    assert_size_stride(arg1091_1, (128, ), (1, ))
    assert_size_stride(arg1092_1, (128, ), (1, ))
    assert_size_stride(arg1093_1, (128, ), (1, ))
    assert_size_stride(arg1094_1, (512, 128), (128, 1))
    assert_size_stride(arg1095_1, (512, ), (1, ))
    assert_size_stride(arg1096_1, (128, 512), (512, 1))
    assert_size_stride(arg1097_1, (128, ), (1, ))
    assert_size_stride(arg1098_1, (128, ), (1, ))
    assert_size_stride(arg1099_1, (128, ), (1, ))
    assert_size_stride(arg1100_1, (512, 128), (128, 1))
    assert_size_stride(arg1101_1, (512, ), (1, ))
    assert_size_stride(arg1102_1, (128, 512), (512, 1))
    assert_size_stride(arg1103_1, (128, ), (1, ))
    assert_size_stride(arg1104_1, (128, ), (1, ))
    assert_size_stride(arg1105_1, (128, ), (1, ))
    assert_size_stride(arg1106_1, (512, 128), (128, 1))
    assert_size_stride(arg1107_1, (512, ), (1, ))
    assert_size_stride(arg1108_1, (128, 512), (512, 1))
    assert_size_stride(arg1109_1, (128, ), (1, ))
    assert_size_stride(arg1110_1, (128, ), (1, ))
    assert_size_stride(arg1111_1, (128, ), (1, ))
    assert_size_stride(arg1112_1, (1, 512), (512, 1))
    assert_size_stride(arg1113_1, (512, 512), (512, 1))
    assert_size_stride(arg1114_1, (512, ), (1, ))
    assert_size_stride(arg1115_1, (512, ), (1, ))
    assert_size_stride(arg1116_1, (512, ), (1, ))
    assert_size_stride(arg1117_1, (384, 30522), (30522, 1))
    assert_size_stride(arg1118_1, (30522, ), (1, ))
    assert_size_stride(arg1119_1, (128, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 384), (49152, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inputs_embeds_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, arg1_1, buf0, 6291456, grid=grid(6291456), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((16384, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf0, (16384, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 512), (1, 384), 0), out=buf1)
        del arg4_1
        del buf0
        buf2 = reinterpret_tensor(buf1, (128, 128, 512), (65536, 512, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [position_embeddings, add, token_type_ids, token_type_embeddings, embeddings, mul_1, embeddings_1], Original ATen: [aten.embedding, aten.add, aten.zeros, aten.mul]
        triton_poi_fused_add_embedding_mul_zeros_1.run(buf2, arg5_1, arg1112_1, arg2_1, arg3_1, arg7_1, arg6_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1112_1
        del arg2_1
        del arg3_1
        del arg5_1
        del arg6_1
        del arg7_1
        buf3 = empty_strided_cuda((16384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2, (16384, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 128), (1, 512), 0), out=buf3)
        del arg32_1
        buf4 = reinterpret_tensor(buf3, (128, 128, 128), (16384, 128, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [mul_3, layer_input_3], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf4, arg33_1, arg35_1, arg34_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        buf5 = empty_strided_cuda((16384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg9_1, reinterpret_tensor(buf4, (16384, 128), (128, 1), 0), reinterpret_tensor(arg8_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
        del arg8_1
        del arg9_1
        buf6 = empty_strided_cuda((16384, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg11_1, reinterpret_tensor(buf4, (16384, 128), (128, 1), 0), reinterpret_tensor(arg10_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg10_1
        del arg11_1
        buf7 = reinterpret_tensor(buf4, (16384, 128), (128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg13_1, reinterpret_tensor(buf2, (16384, 512), (512, 1), 0), reinterpret_tensor(arg12_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf7)
        del arg12_1
        del arg13_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf6, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf7, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf5
        buf9 = buf8[0]
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf9, (16384, 128), (128, 1), 0), reinterpret_tensor(arg14_1, (128, 128), (1, 128), 0), out=buf13)
        del arg14_1
        buf14 = reinterpret_tensor(buf9, (16384, 128), (128, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2, (16384, 512), (512, 1), 0), reinterpret_tensor(arg28_1, (512, 128), (1, 512), 0), out=buf14)
        del arg28_1
        buf15 = reinterpret_tensor(buf13, (128, 128, 128), (16384, 128, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [mul_2, layer_input_1, add_6, mul_4, layer_outputs_1], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf15, arg15_1, buf14, arg29_1, arg31_1, arg30_1, arg17_1, arg16_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg15_1
        del arg16_1
        del arg17_1
        del arg29_1
        del arg30_1
        del arg31_1
        buf16 = empty_strided_cuda((16384, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf15, (16384, 128), (128, 1), 0), reinterpret_tensor(arg36_1, (128, 512), (1, 128), 0), out=buf16)
        del arg36_1
        buf17 = reinterpret_tensor(buf16, (128, 128, 512), (65536, 512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf17, arg37_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg37_1
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf17, (16384, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 128), (1, 512), 0), out=buf18)
        del arg38_1
        buf19 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [add_8, mul_5, layer_outputs_3], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf19, buf18, arg39_1, arg41_1, arg40_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        buf20 = reinterpret_tensor(buf17, (16384, 512), (512, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (16384, 128), (128, 1), 0), reinterpret_tensor(arg42_1, (128, 512), (1, 128), 0), out=buf20)
        del arg42_1
        buf21 = reinterpret_tensor(buf20, (128, 128, 512), (65536, 512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf21, arg43_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg43_1
        buf22 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf21, (16384, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 128), (1, 512), 0), out=buf22)
        del arg44_1
        buf23 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [add_10, mul_6, layer_outputs_5], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf23, buf22, arg45_1, arg47_1, arg46_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        buf24 = reinterpret_tensor(buf21, (16384, 512), (512, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf23, (16384, 128), (128, 1), 0), reinterpret_tensor(arg48_1, (128, 512), (1, 128), 0), out=buf24)
        del arg48_1
        buf25 = reinterpret_tensor(buf24, (128, 128, 512), (65536, 512, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf25, arg49_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg49_1
        buf26 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf25, (16384, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 128), (1, 512), 0), out=buf26)
        del arg50_1
        buf27 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [add_12, mul_7, layer_outputs_7], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf27, buf26, arg51_1, arg53_1, arg52_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        buf28 = reinterpret_tensor(buf25, (16384, 512), (512, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf27, (16384, 128), (128, 1), 0), reinterpret_tensor(arg18_1, (128, 512), (1, 128), 0), out=buf28)
        del arg18_1
        buf29 = reinterpret_tensor(buf28, (128, 128, 512), (65536, 512, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_7], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf29, arg19_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg19_1
        buf30 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (16384, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 128), (1, 512), 0), out=buf30)
        del arg20_1
        buf31 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [add_14, mul_8, layer_output_1], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf31, buf30, arg21_1, arg23_1, arg22_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        buf32 = reinterpret_tensor(buf29, (16384, 512), (512, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf31, (16384, 128), (128, 1), 0), reinterpret_tensor(arg24_1, (128, 512), (1, 128), 0), out=buf32)
        del arg24_1
        buf33 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [add_16, mul_9, layer_outputs_10], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf33, buf32, arg25_1, arg27_1, arg26_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        buf34 = reinterpret_tensor(buf31, (16384, 128), (128, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (16384, 512), (512, 1), 0), reinterpret_tensor(arg78_1, (512, 128), (1, 512), 0), out=buf34)
        del arg78_1
        buf35 = reinterpret_tensor(buf34, (128, 128, 128), (16384, 128, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [mul_11, layer_input_7], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf35, arg79_1, arg81_1, arg80_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg79_1
        del arg80_1
        del arg81_1
        buf36 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf35, (16384, 128), (128, 1), 0), reinterpret_tensor(arg54_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf36)
        del arg54_1
        del arg55_1
        buf37 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg57_1, reinterpret_tensor(buf35, (16384, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf37)
        del arg56_1
        del arg57_1
        buf38 = reinterpret_tensor(buf35, (16384, 128), (128, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg59_1, reinterpret_tensor(buf33, (16384, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf38)
        del arg58_1
        del arg59_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf39 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf36, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf37, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf38, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf36
        buf40 = buf39[0]
        del buf39
        buf44 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf40, (16384, 128), (128, 1), 0), reinterpret_tensor(arg60_1, (128, 128), (1, 128), 0), out=buf44)
        del arg60_1
        buf45 = reinterpret_tensor(buf40, (16384, 128), (128, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf33, (16384, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 128), (1, 512), 0), out=buf45)
        del arg74_1
        buf46 = reinterpret_tensor(buf44, (128, 128, 128), (16384, 128, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [mul_10, layer_input_5, add_21, mul_12, layer_outputs_12], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf46, arg61_1, buf45, arg75_1, arg77_1, arg76_1, arg63_1, arg62_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg75_1
        del arg76_1
        del arg77_1
        buf47 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (16384, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 512), (1, 128), 0), out=buf47)
        del arg82_1
        buf48 = reinterpret_tensor(buf47, (128, 128, 512), (65536, 512, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf48, arg83_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg83_1
        buf49 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf48, (16384, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 128), (1, 512), 0), out=buf49)
        del arg84_1
        buf50 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [add_23, mul_13, layer_outputs_14], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf50, buf49, arg85_1, arg87_1, arg86_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg85_1
        del arg86_1
        del arg87_1
        buf51 = reinterpret_tensor(buf48, (16384, 512), (512, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (16384, 128), (128, 1), 0), reinterpret_tensor(arg88_1, (128, 512), (1, 128), 0), out=buf51)
        del arg88_1
        buf52 = reinterpret_tensor(buf51, (128, 128, 512), (65536, 512, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf52, arg89_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg89_1
        buf53 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (16384, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 128), (1, 512), 0), out=buf53)
        del arg90_1
        buf54 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [add_25, mul_14, layer_outputs_16], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf54, buf53, arg91_1, arg93_1, arg92_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        buf55 = reinterpret_tensor(buf52, (16384, 512), (512, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf54, (16384, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 512), (1, 128), 0), out=buf55)
        del arg94_1
        buf56 = reinterpret_tensor(buf55, (128, 128, 512), (65536, 512, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_13], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf56, arg95_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg95_1
        buf57 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (16384, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 128), (1, 512), 0), out=buf57)
        del arg96_1
        buf58 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [add_27, mul_15, layer_outputs_18], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf58, buf57, arg97_1, arg99_1, arg98_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg97_1
        del arg98_1
        del arg99_1
        buf59 = reinterpret_tensor(buf56, (16384, 512), (512, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (16384, 128), (128, 1), 0), reinterpret_tensor(arg64_1, (128, 512), (1, 128), 0), out=buf59)
        del arg64_1
        buf60 = reinterpret_tensor(buf59, (128, 128, 512), (65536, 512, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf60, arg65_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg65_1
        buf61 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (16384, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 128), (1, 512), 0), out=buf61)
        del arg66_1
        buf62 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [add_29, mul_16, layer_output_3], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf62, buf61, arg67_1, arg69_1, arg68_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        buf63 = reinterpret_tensor(buf60, (16384, 512), (512, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf62, (16384, 128), (128, 1), 0), reinterpret_tensor(arg70_1, (128, 512), (1, 128), 0), out=buf63)
        del arg70_1
        buf64 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [add_31, mul_17, layer_outputs_21], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf64, buf63, arg71_1, arg73_1, arg72_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg71_1
        del arg72_1
        del arg73_1
        buf65 = reinterpret_tensor(buf62, (16384, 128), (128, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (16384, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 128), (1, 512), 0), out=buf65)
        del arg124_1
        buf66 = reinterpret_tensor(buf65, (128, 128, 128), (16384, 128, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [mul_19, layer_input_11], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf66, arg125_1, arg127_1, arg126_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        buf67 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg101_1, reinterpret_tensor(buf66, (16384, 128), (128, 1), 0), reinterpret_tensor(arg100_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf67)
        del arg100_1
        del arg101_1
        buf68 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg103_1, reinterpret_tensor(buf66, (16384, 128), (128, 1), 0), reinterpret_tensor(arg102_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf68)
        del arg102_1
        del arg103_1
        buf69 = reinterpret_tensor(buf66, (16384, 128), (128, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf64, (16384, 512), (512, 1), 0), reinterpret_tensor(arg104_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf69)
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf70 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf67, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf68, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf69, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf67
        buf71 = buf70[0]
        del buf70
        buf75 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (16384, 128), (128, 1), 0), reinterpret_tensor(arg106_1, (128, 128), (1, 128), 0), out=buf75)
        del arg106_1
        buf76 = reinterpret_tensor(buf71, (16384, 128), (128, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (16384, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 128), (1, 512), 0), out=buf76)
        del arg120_1
        buf77 = reinterpret_tensor(buf75, (128, 128, 128), (16384, 128, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [mul_18, layer_input_9, add_36, mul_20, layer_outputs_23], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf77, arg107_1, buf76, arg121_1, arg123_1, arg122_1, arg109_1, arg108_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg121_1
        del arg122_1
        del arg123_1
        buf78 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (16384, 128), (128, 1), 0), reinterpret_tensor(arg128_1, (128, 512), (1, 128), 0), out=buf78)
        del arg128_1
        buf79 = reinterpret_tensor(buf78, (128, 128, 512), (65536, 512, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf79, arg129_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg129_1
        buf80 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (16384, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 128), (1, 512), 0), out=buf80)
        del arg130_1
        buf81 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [add_38, mul_21, layer_outputs_25], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf81, buf80, arg131_1, arg133_1, arg132_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        buf82 = reinterpret_tensor(buf79, (16384, 512), (512, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (16384, 128), (128, 1), 0), reinterpret_tensor(arg134_1, (128, 512), (1, 128), 0), out=buf82)
        del arg134_1
        buf83 = reinterpret_tensor(buf82, (128, 128, 512), (65536, 512, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf83, arg135_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg135_1
        buf84 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (16384, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 128), (1, 512), 0), out=buf84)
        del arg136_1
        buf85 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [add_40, mul_22, layer_outputs_27], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf85, buf84, arg137_1, arg139_1, arg138_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        buf86 = reinterpret_tensor(buf83, (16384, 512), (512, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (16384, 128), (128, 1), 0), reinterpret_tensor(arg140_1, (128, 512), (1, 128), 0), out=buf86)
        del arg140_1
        buf87 = reinterpret_tensor(buf86, (128, 128, 512), (65536, 512, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf87, arg141_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg141_1
        buf88 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (16384, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 128), (1, 512), 0), out=buf88)
        del arg142_1
        buf89 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [add_42, mul_23, layer_outputs_29], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf89, buf88, arg143_1, arg145_1, arg144_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg143_1
        del arg144_1
        del arg145_1
        buf90 = reinterpret_tensor(buf87, (16384, 512), (512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf89, (16384, 128), (128, 1), 0), reinterpret_tensor(arg110_1, (128, 512), (1, 128), 0), out=buf90)
        del arg110_1
        buf91 = reinterpret_tensor(buf90, (128, 128, 512), (65536, 512, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_23], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf91, arg111_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg111_1
        buf92 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf91, (16384, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 128), (1, 512), 0), out=buf92)
        del arg112_1
        buf93 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [add_44, mul_24, layer_output_5], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf93, buf92, arg113_1, arg115_1, arg114_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        buf94 = reinterpret_tensor(buf91, (16384, 512), (512, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (16384, 128), (128, 1), 0), reinterpret_tensor(arg116_1, (128, 512), (1, 128), 0), out=buf94)
        del arg116_1
        buf95 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [add_46, mul_25, layer_outputs_32], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf95, buf94, arg117_1, arg119_1, arg118_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        buf96 = reinterpret_tensor(buf93, (16384, 128), (128, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (16384, 512), (512, 1), 0), reinterpret_tensor(arg170_1, (512, 128), (1, 512), 0), out=buf96)
        del arg170_1
        buf97 = reinterpret_tensor(buf96, (128, 128, 128), (16384, 128, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [mul_27, layer_input_15], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf97, arg171_1, arg173_1, arg172_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg171_1
        del arg172_1
        del arg173_1
        buf98 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf97, (16384, 128), (128, 1), 0), reinterpret_tensor(arg146_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf98)
        del arg146_1
        del arg147_1
        buf99 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg149_1, reinterpret_tensor(buf97, (16384, 128), (128, 1), 0), reinterpret_tensor(arg148_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf99)
        del arg148_1
        del arg149_1
        buf100 = reinterpret_tensor(buf97, (16384, 128), (128, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg151_1, reinterpret_tensor(buf95, (16384, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf100)
        del arg150_1
        del arg151_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf101 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf98, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf99, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf100, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf100
        buf102 = buf101[0]
        del buf101
        buf106 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (16384, 128), (128, 1), 0), reinterpret_tensor(arg152_1, (128, 128), (1, 128), 0), out=buf106)
        del arg152_1
        buf107 = reinterpret_tensor(buf102, (16384, 128), (128, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (16384, 512), (512, 1), 0), reinterpret_tensor(arg166_1, (512, 128), (1, 512), 0), out=buf107)
        del arg166_1
        buf108 = reinterpret_tensor(buf106, (128, 128, 128), (16384, 128, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [mul_26, layer_input_13, add_51, mul_28, layer_outputs_34], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf108, arg153_1, buf107, arg167_1, arg169_1, arg168_1, arg155_1, arg154_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        del arg167_1
        del arg168_1
        del arg169_1
        buf109 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf108, (16384, 128), (128, 1), 0), reinterpret_tensor(arg174_1, (128, 512), (1, 128), 0), out=buf109)
        del arg174_1
        buf110 = reinterpret_tensor(buf109, (128, 128, 512), (65536, 512, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf110, arg175_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg175_1
        buf111 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf110, (16384, 512), (512, 1), 0), reinterpret_tensor(arg176_1, (512, 128), (1, 512), 0), out=buf111)
        del arg176_1
        buf112 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [add_53, mul_29, layer_outputs_36], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf112, buf111, arg177_1, arg179_1, arg178_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        buf113 = reinterpret_tensor(buf110, (16384, 512), (512, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf112, (16384, 128), (128, 1), 0), reinterpret_tensor(arg180_1, (128, 512), (1, 128), 0), out=buf113)
        del arg180_1
        buf114 = reinterpret_tensor(buf113, (128, 128, 512), (65536, 512, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf114, arg181_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg181_1
        buf115 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf114, (16384, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 128), (1, 512), 0), out=buf115)
        del arg182_1
        buf116 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [add_55, mul_30, layer_outputs_38], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf116, buf115, arg183_1, arg185_1, arg184_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        buf117 = reinterpret_tensor(buf114, (16384, 512), (512, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf116, (16384, 128), (128, 1), 0), reinterpret_tensor(arg186_1, (128, 512), (1, 128), 0), out=buf117)
        del arg186_1
        buf118 = reinterpret_tensor(buf117, (128, 128, 512), (65536, 512, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf118, arg187_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg187_1
        buf119 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf118, (16384, 512), (512, 1), 0), reinterpret_tensor(arg188_1, (512, 128), (1, 512), 0), out=buf119)
        del arg188_1
        buf120 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [add_57, mul_31, layer_outputs_40], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf120, buf119, arg189_1, arg191_1, arg190_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg189_1
        del arg190_1
        del arg191_1
        buf121 = reinterpret_tensor(buf118, (16384, 512), (512, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf120, (16384, 128), (128, 1), 0), reinterpret_tensor(arg156_1, (128, 512), (1, 128), 0), out=buf121)
        del arg156_1
        buf122 = reinterpret_tensor(buf121, (128, 128, 512), (65536, 512, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf122, arg157_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg157_1
        buf123 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (16384, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 128), (1, 512), 0), out=buf123)
        del arg158_1
        buf124 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [add_59, mul_32, layer_output_7], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf124, buf123, arg159_1, arg161_1, arg160_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        buf125 = reinterpret_tensor(buf122, (16384, 512), (512, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (16384, 128), (128, 1), 0), reinterpret_tensor(arg162_1, (128, 512), (1, 128), 0), out=buf125)
        del arg162_1
        buf126 = reinterpret_tensor(buf125, (128, 128, 512), (65536, 512, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [add_61, mul_33, layer_outputs_43], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_7.run(buf126, arg163_1, buf95, arg165_1, arg164_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        buf127 = reinterpret_tensor(buf124, (16384, 128), (128, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (16384, 512), (512, 1), 0), reinterpret_tensor(arg216_1, (512, 128), (1, 512), 0), out=buf127)
        del arg216_1
        buf128 = reinterpret_tensor(buf127, (128, 128, 128), (16384, 128, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [mul_35, layer_input_19], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf128, arg217_1, arg219_1, arg218_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        buf129 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg193_1, reinterpret_tensor(buf128, (16384, 128), (128, 1), 0), reinterpret_tensor(arg192_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf129)
        del arg192_1
        del arg193_1
        buf130 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg195_1, reinterpret_tensor(buf128, (16384, 128), (128, 1), 0), reinterpret_tensor(arg194_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf130)
        del arg194_1
        del arg195_1
        buf131 = reinterpret_tensor(buf128, (16384, 128), (128, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg197_1, reinterpret_tensor(buf126, (16384, 512), (512, 1), 0), reinterpret_tensor(arg196_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf131)
        del arg196_1
        del arg197_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf129, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf130, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf131, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf129
        buf133 = buf132[0]
        del buf132
        buf137 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf133, (16384, 128), (128, 1), 0), reinterpret_tensor(arg198_1, (128, 128), (1, 128), 0), out=buf137)
        del arg198_1
        buf138 = reinterpret_tensor(buf133, (16384, 128), (128, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf126, (16384, 512), (512, 1), 0), reinterpret_tensor(arg212_1, (512, 128), (1, 512), 0), out=buf138)
        del arg212_1
        buf139 = reinterpret_tensor(buf137, (128, 128, 128), (16384, 128, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [mul_34, layer_input_17, add_66, mul_36, layer_outputs_45], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf139, arg199_1, buf138, arg213_1, arg215_1, arg214_1, arg201_1, arg200_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg199_1
        del arg200_1
        del arg201_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf140 = reinterpret_tensor(buf95, (16384, 512), (512, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf139, (16384, 128), (128, 1), 0), reinterpret_tensor(arg220_1, (128, 512), (1, 128), 0), out=buf140)
        del arg220_1
        buf141 = reinterpret_tensor(buf140, (128, 128, 512), (65536, 512, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_33], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf141, arg221_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg221_1
        buf142 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (16384, 512), (512, 1), 0), reinterpret_tensor(arg222_1, (512, 128), (1, 512), 0), out=buf142)
        del arg222_1
        buf143 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [add_68, mul_37, layer_outputs_47], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf143, buf142, arg223_1, arg225_1, arg224_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg223_1
        del arg224_1
        del arg225_1
        buf144 = reinterpret_tensor(buf141, (16384, 512), (512, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (16384, 128), (128, 1), 0), reinterpret_tensor(arg226_1, (128, 512), (1, 128), 0), out=buf144)
        del arg226_1
        buf145 = reinterpret_tensor(buf144, (128, 128, 512), (65536, 512, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf145, arg227_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg227_1
        buf146 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (16384, 512), (512, 1), 0), reinterpret_tensor(arg228_1, (512, 128), (1, 512), 0), out=buf146)
        del arg228_1
        buf147 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [add_70, mul_38, layer_outputs_49], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf147, buf146, arg229_1, arg231_1, arg230_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg229_1
        del arg230_1
        del arg231_1
        buf148 = reinterpret_tensor(buf145, (16384, 512), (512, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (16384, 128), (128, 1), 0), reinterpret_tensor(arg232_1, (128, 512), (1, 128), 0), out=buf148)
        del arg232_1
        buf149 = reinterpret_tensor(buf148, (128, 128, 512), (65536, 512, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_37], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf149, arg233_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg233_1
        buf150 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (16384, 512), (512, 1), 0), reinterpret_tensor(arg234_1, (512, 128), (1, 512), 0), out=buf150)
        del arg234_1
        buf151 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [add_72, mul_39, layer_outputs_51], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf151, buf150, arg235_1, arg237_1, arg236_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg235_1
        del arg236_1
        del arg237_1
        buf152 = reinterpret_tensor(buf149, (16384, 512), (512, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf151, (16384, 128), (128, 1), 0), reinterpret_tensor(arg202_1, (128, 512), (1, 128), 0), out=buf152)
        del arg202_1
        buf153 = reinterpret_tensor(buf152, (128, 128, 512), (65536, 512, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf153, arg203_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg203_1
        buf154 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (16384, 512), (512, 1), 0), reinterpret_tensor(arg204_1, (512, 128), (1, 512), 0), out=buf154)
        del arg204_1
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [add_74, mul_40, layer_output_9], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf155, buf154, arg205_1, arg207_1, arg206_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        buf156 = reinterpret_tensor(buf153, (16384, 512), (512, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (16384, 128), (128, 1), 0), reinterpret_tensor(arg208_1, (128, 512), (1, 128), 0), out=buf156)
        del arg208_1
        buf157 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [add_76, mul_41, layer_outputs_54], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf157, buf156, arg209_1, arg211_1, arg210_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg209_1
        del arg210_1
        del arg211_1
        buf158 = reinterpret_tensor(buf155, (16384, 128), (128, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (16384, 512), (512, 1), 0), reinterpret_tensor(arg262_1, (512, 128), (1, 512), 0), out=buf158)
        del arg262_1
        buf159 = reinterpret_tensor(buf158, (128, 128, 128), (16384, 128, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [mul_43, layer_input_23], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf159, arg263_1, arg265_1, arg264_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg263_1
        del arg264_1
        del arg265_1
        buf160 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg239_1, reinterpret_tensor(buf159, (16384, 128), (128, 1), 0), reinterpret_tensor(arg238_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf160)
        del arg238_1
        del arg239_1
        buf161 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg241_1, reinterpret_tensor(buf159, (16384, 128), (128, 1), 0), reinterpret_tensor(arg240_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf161)
        del arg240_1
        del arg241_1
        buf162 = reinterpret_tensor(buf159, (16384, 128), (128, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg243_1, reinterpret_tensor(buf157, (16384, 512), (512, 1), 0), reinterpret_tensor(arg242_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf162)
        del arg242_1
        del arg243_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf163 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf160, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf161, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf162, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf160
        buf164 = buf163[0]
        del buf163
        buf168 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (16384, 128), (128, 1), 0), reinterpret_tensor(arg244_1, (128, 128), (1, 128), 0), out=buf168)
        del arg244_1
        buf169 = reinterpret_tensor(buf164, (16384, 128), (128, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (16384, 512), (512, 1), 0), reinterpret_tensor(arg258_1, (512, 128), (1, 512), 0), out=buf169)
        del arg258_1
        buf170 = reinterpret_tensor(buf168, (128, 128, 128), (16384, 128, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [mul_42, layer_input_21, add_81, mul_44, layer_outputs_56], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf170, arg245_1, buf169, arg259_1, arg261_1, arg260_1, arg247_1, arg246_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg245_1
        del arg246_1
        del arg247_1
        del arg259_1
        del arg260_1
        del arg261_1
        buf171 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (16384, 128), (128, 1), 0), reinterpret_tensor(arg266_1, (128, 512), (1, 128), 0), out=buf171)
        del arg266_1
        buf172 = reinterpret_tensor(buf171, (128, 128, 512), (65536, 512, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_41], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf172, arg267_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg267_1
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf172, (16384, 512), (512, 1), 0), reinterpret_tensor(arg268_1, (512, 128), (1, 512), 0), out=buf173)
        del arg268_1
        buf174 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [add_83, mul_45, layer_outputs_58], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf174, buf173, arg269_1, arg271_1, arg270_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        buf175 = reinterpret_tensor(buf172, (16384, 512), (512, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (16384, 128), (128, 1), 0), reinterpret_tensor(arg272_1, (128, 512), (1, 128), 0), out=buf175)
        del arg272_1
        buf176 = reinterpret_tensor(buf175, (128, 128, 512), (65536, 512, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf176, arg273_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg273_1
        buf177 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (16384, 512), (512, 1), 0), reinterpret_tensor(arg274_1, (512, 128), (1, 512), 0), out=buf177)
        del arg274_1
        buf178 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [add_85, mul_46, layer_outputs_60], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf178, buf177, arg275_1, arg277_1, arg276_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg275_1
        del arg276_1
        del arg277_1
        buf179 = reinterpret_tensor(buf176, (16384, 512), (512, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf178, (16384, 128), (128, 1), 0), reinterpret_tensor(arg278_1, (128, 512), (1, 128), 0), out=buf179)
        del arg278_1
        buf180 = reinterpret_tensor(buf179, (128, 128, 512), (65536, 512, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_45], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf180, arg279_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg279_1
        buf181 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (16384, 512), (512, 1), 0), reinterpret_tensor(arg280_1, (512, 128), (1, 512), 0), out=buf181)
        del arg280_1
        buf182 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [add_87, mul_47, layer_outputs_62], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf182, buf181, arg281_1, arg283_1, arg282_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg281_1
        del arg282_1
        del arg283_1
        buf183 = reinterpret_tensor(buf180, (16384, 512), (512, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (16384, 128), (128, 1), 0), reinterpret_tensor(arg248_1, (128, 512), (1, 128), 0), out=buf183)
        del arg248_1
        buf184 = reinterpret_tensor(buf183, (128, 128, 512), (65536, 512, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_47], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf184, arg249_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg249_1
        buf185 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (16384, 512), (512, 1), 0), reinterpret_tensor(arg250_1, (512, 128), (1, 512), 0), out=buf185)
        del arg250_1
        buf186 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [add_89, mul_48, layer_output_11], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf186, buf185, arg251_1, arg253_1, arg252_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        buf187 = reinterpret_tensor(buf184, (16384, 512), (512, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (16384, 128), (128, 1), 0), reinterpret_tensor(arg254_1, (128, 512), (1, 128), 0), out=buf187)
        del arg254_1
        buf188 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [add_91, mul_49, layer_outputs_65], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf188, buf187, arg255_1, arg257_1, arg256_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg255_1
        del arg256_1
        del arg257_1
        buf189 = reinterpret_tensor(buf186, (16384, 128), (128, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (16384, 512), (512, 1), 0), reinterpret_tensor(arg308_1, (512, 128), (1, 512), 0), out=buf189)
        del arg308_1
        buf190 = reinterpret_tensor(buf189, (128, 128, 128), (16384, 128, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [mul_51, layer_input_27], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf190, arg309_1, arg311_1, arg310_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg309_1
        del arg310_1
        del arg311_1
        buf191 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg285_1, reinterpret_tensor(buf190, (16384, 128), (128, 1), 0), reinterpret_tensor(arg284_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf191)
        del arg284_1
        del arg285_1
        buf192 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg287_1, reinterpret_tensor(buf190, (16384, 128), (128, 1), 0), reinterpret_tensor(arg286_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf192)
        del arg286_1
        del arg287_1
        buf193 = reinterpret_tensor(buf190, (16384, 128), (128, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg289_1, reinterpret_tensor(buf188, (16384, 512), (512, 1), 0), reinterpret_tensor(arg288_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf193)
        del arg288_1
        del arg289_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf194 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf191, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf192, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf193, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf191
        buf195 = buf194[0]
        del buf194
        buf199 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf195, (16384, 128), (128, 1), 0), reinterpret_tensor(arg290_1, (128, 128), (1, 128), 0), out=buf199)
        del arg290_1
        buf200 = reinterpret_tensor(buf195, (16384, 128), (128, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (16384, 512), (512, 1), 0), reinterpret_tensor(arg304_1, (512, 128), (1, 512), 0), out=buf200)
        del arg304_1
        buf201 = reinterpret_tensor(buf199, (128, 128, 128), (16384, 128, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [mul_50, layer_input_25, add_96, mul_52, layer_outputs_67], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf201, arg291_1, buf200, arg305_1, arg307_1, arg306_1, arg293_1, arg292_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg291_1
        del arg292_1
        del arg293_1
        del arg305_1
        del arg306_1
        del arg307_1
        buf202 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf201, (16384, 128), (128, 1), 0), reinterpret_tensor(arg312_1, (128, 512), (1, 128), 0), out=buf202)
        del arg312_1
        buf203 = reinterpret_tensor(buf202, (128, 128, 512), (65536, 512, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_49], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf203, arg313_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg313_1
        buf204 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf203, (16384, 512), (512, 1), 0), reinterpret_tensor(arg314_1, (512, 128), (1, 512), 0), out=buf204)
        del arg314_1
        buf205 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [add_98, mul_53, layer_outputs_69], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf205, buf204, arg315_1, arg317_1, arg316_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg315_1
        del arg316_1
        del arg317_1
        buf206 = reinterpret_tensor(buf203, (16384, 512), (512, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf205, (16384, 128), (128, 1), 0), reinterpret_tensor(arg318_1, (128, 512), (1, 128), 0), out=buf206)
        del arg318_1
        buf207 = reinterpret_tensor(buf206, (128, 128, 512), (65536, 512, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf207, arg319_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg319_1
        buf208 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (16384, 512), (512, 1), 0), reinterpret_tensor(arg320_1, (512, 128), (1, 512), 0), out=buf208)
        del arg320_1
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [add_100, mul_54, layer_outputs_71], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf209, buf208, arg321_1, arg323_1, arg322_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg321_1
        del arg322_1
        del arg323_1
        buf210 = reinterpret_tensor(buf207, (16384, 512), (512, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf209, (16384, 128), (128, 1), 0), reinterpret_tensor(arg324_1, (128, 512), (1, 128), 0), out=buf210)
        del arg324_1
        buf211 = reinterpret_tensor(buf210, (128, 128, 512), (65536, 512, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf211, arg325_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg325_1
        buf212 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (16384, 512), (512, 1), 0), reinterpret_tensor(arg326_1, (512, 128), (1, 512), 0), out=buf212)
        del arg326_1
        buf213 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [add_102, mul_55, layer_outputs_73], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf213, buf212, arg327_1, arg329_1, arg328_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        buf214 = reinterpret_tensor(buf211, (16384, 512), (512, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf213, (16384, 128), (128, 1), 0), reinterpret_tensor(arg294_1, (128, 512), (1, 128), 0), out=buf214)
        del arg294_1
        buf215 = reinterpret_tensor(buf214, (128, 128, 512), (65536, 512, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf215, arg295_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg295_1
        buf216 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf215, (16384, 512), (512, 1), 0), reinterpret_tensor(arg296_1, (512, 128), (1, 512), 0), out=buf216)
        del arg296_1
        buf217 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [add_104, mul_56, layer_output_13], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf217, buf216, arg297_1, arg299_1, arg298_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        buf218 = reinterpret_tensor(buf215, (16384, 512), (512, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf217, (16384, 128), (128, 1), 0), reinterpret_tensor(arg300_1, (128, 512), (1, 128), 0), out=buf218)
        del arg300_1
        buf219 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [add_106, mul_57, layer_outputs_76], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf219, buf218, arg301_1, arg303_1, arg302_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg301_1
        del arg302_1
        del arg303_1
        buf220 = reinterpret_tensor(buf217, (16384, 128), (128, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (16384, 512), (512, 1), 0), reinterpret_tensor(arg354_1, (512, 128), (1, 512), 0), out=buf220)
        del arg354_1
        buf221 = reinterpret_tensor(buf220, (128, 128, 128), (16384, 128, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [mul_59, layer_input_31], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf221, arg355_1, arg357_1, arg356_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg355_1
        del arg356_1
        del arg357_1
        buf222 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg331_1, reinterpret_tensor(buf221, (16384, 128), (128, 1), 0), reinterpret_tensor(arg330_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf222)
        del arg330_1
        del arg331_1
        buf223 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg333_1, reinterpret_tensor(buf221, (16384, 128), (128, 1), 0), reinterpret_tensor(arg332_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf223)
        del arg332_1
        del arg333_1
        buf224 = reinterpret_tensor(buf221, (16384, 128), (128, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg335_1, reinterpret_tensor(buf219, (16384, 512), (512, 1), 0), reinterpret_tensor(arg334_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf224)
        del arg334_1
        del arg335_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf225 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf222, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf223, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf224, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf222
        buf226 = buf225[0]
        del buf225
        buf230 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (16384, 128), (128, 1), 0), reinterpret_tensor(arg336_1, (128, 128), (1, 128), 0), out=buf230)
        del arg336_1
        buf231 = reinterpret_tensor(buf226, (16384, 128), (128, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf219, (16384, 512), (512, 1), 0), reinterpret_tensor(arg350_1, (512, 128), (1, 512), 0), out=buf231)
        del arg350_1
        buf232 = reinterpret_tensor(buf230, (128, 128, 128), (16384, 128, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [mul_58, layer_input_29, add_111, mul_60, layer_outputs_78], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf232, arg337_1, buf231, arg351_1, arg353_1, arg352_1, arg339_1, arg338_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg351_1
        del arg352_1
        del arg353_1
        buf233 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (16384, 128), (128, 1), 0), reinterpret_tensor(arg358_1, (128, 512), (1, 128), 0), out=buf233)
        del arg358_1
        buf234 = reinterpret_tensor(buf233, (128, 128, 512), (65536, 512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_57], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf234, arg359_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg359_1
        buf235 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (16384, 512), (512, 1), 0), reinterpret_tensor(arg360_1, (512, 128), (1, 512), 0), out=buf235)
        del arg360_1
        buf236 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [add_113, mul_61, layer_outputs_80], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf236, buf235, arg361_1, arg363_1, arg362_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg361_1
        del arg362_1
        del arg363_1
        buf237 = reinterpret_tensor(buf234, (16384, 512), (512, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf236, (16384, 128), (128, 1), 0), reinterpret_tensor(arg364_1, (128, 512), (1, 128), 0), out=buf237)
        del arg364_1
        buf238 = reinterpret_tensor(buf237, (128, 128, 512), (65536, 512, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf238, arg365_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg365_1
        buf239 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf238, (16384, 512), (512, 1), 0), reinterpret_tensor(arg366_1, (512, 128), (1, 512), 0), out=buf239)
        del arg366_1
        buf240 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [add_115, mul_62, layer_outputs_82], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf240, buf239, arg367_1, arg369_1, arg368_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        buf241 = reinterpret_tensor(buf238, (16384, 512), (512, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (16384, 128), (128, 1), 0), reinterpret_tensor(arg370_1, (128, 512), (1, 128), 0), out=buf241)
        del arg370_1
        buf242 = reinterpret_tensor(buf241, (128, 128, 512), (65536, 512, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf242, arg371_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg371_1
        buf243 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (16384, 512), (512, 1), 0), reinterpret_tensor(arg372_1, (512, 128), (1, 512), 0), out=buf243)
        del arg372_1
        buf244 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [add_117, mul_63, layer_outputs_84], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf244, buf243, arg373_1, arg375_1, arg374_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg373_1
        del arg374_1
        del arg375_1
        buf245 = reinterpret_tensor(buf242, (16384, 512), (512, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (16384, 128), (128, 1), 0), reinterpret_tensor(arg340_1, (128, 512), (1, 128), 0), out=buf245)
        del arg340_1
        buf246 = reinterpret_tensor(buf245, (128, 128, 512), (65536, 512, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_63], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf246, arg341_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg341_1
        buf247 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf246, (16384, 512), (512, 1), 0), reinterpret_tensor(arg342_1, (512, 128), (1, 512), 0), out=buf247)
        del arg342_1
        buf248 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [add_119, mul_64, layer_output_15], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf248, buf247, arg343_1, arg345_1, arg344_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg343_1
        del arg344_1
        del arg345_1
        buf249 = reinterpret_tensor(buf246, (16384, 512), (512, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf248, (16384, 128), (128, 1), 0), reinterpret_tensor(arg346_1, (128, 512), (1, 128), 0), out=buf249)
        del arg346_1
        buf250 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [add_121, mul_65, layer_outputs_87], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf250, buf249, arg347_1, arg349_1, arg348_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        buf251 = reinterpret_tensor(buf248, (16384, 128), (128, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (16384, 512), (512, 1), 0), reinterpret_tensor(arg400_1, (512, 128), (1, 512), 0), out=buf251)
        del arg400_1
        buf252 = reinterpret_tensor(buf251, (128, 128, 128), (16384, 128, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [mul_67, layer_input_35], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf252, arg401_1, arg403_1, arg402_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg401_1
        del arg402_1
        del arg403_1
        buf253 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg377_1, reinterpret_tensor(buf252, (16384, 128), (128, 1), 0), reinterpret_tensor(arg376_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf253)
        del arg376_1
        del arg377_1
        buf254 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg379_1, reinterpret_tensor(buf252, (16384, 128), (128, 1), 0), reinterpret_tensor(arg378_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf254)
        del arg378_1
        del arg379_1
        buf255 = reinterpret_tensor(buf252, (16384, 128), (128, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg381_1, reinterpret_tensor(buf250, (16384, 512), (512, 1), 0), reinterpret_tensor(arg380_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf255)
        del arg380_1
        del arg381_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf256 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf253, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf254, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf255, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf253
        buf257 = buf256[0]
        del buf256
        buf261 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf257, (16384, 128), (128, 1), 0), reinterpret_tensor(arg382_1, (128, 128), (1, 128), 0), out=buf261)
        del arg382_1
        buf262 = reinterpret_tensor(buf257, (16384, 128), (128, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (16384, 512), (512, 1), 0), reinterpret_tensor(arg396_1, (512, 128), (1, 512), 0), out=buf262)
        del arg396_1
        buf263 = reinterpret_tensor(buf261, (128, 128, 128), (16384, 128, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [mul_66, layer_input_33, add_126, mul_68, layer_outputs_89], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf263, arg383_1, buf262, arg397_1, arg399_1, arg398_1, arg385_1, arg384_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg383_1
        del arg384_1
        del arg385_1
        del arg397_1
        del arg398_1
        del arg399_1
        buf264 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf263, (16384, 128), (128, 1), 0), reinterpret_tensor(arg404_1, (128, 512), (1, 128), 0), out=buf264)
        del arg404_1
        buf265 = reinterpret_tensor(buf264, (128, 128, 512), (65536, 512, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf265, arg405_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg405_1
        buf266 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (16384, 512), (512, 1), 0), reinterpret_tensor(arg406_1, (512, 128), (1, 512), 0), out=buf266)
        del arg406_1
        buf267 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [add_128, mul_69, layer_outputs_91], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf267, buf266, arg407_1, arg409_1, arg408_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        buf268 = reinterpret_tensor(buf265, (16384, 512), (512, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (16384, 128), (128, 1), 0), reinterpret_tensor(arg410_1, (128, 512), (1, 128), 0), out=buf268)
        del arg410_1
        buf269 = reinterpret_tensor(buf268, (128, 128, 512), (65536, 512, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf269, arg411_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg411_1
        buf270 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (16384, 512), (512, 1), 0), reinterpret_tensor(arg412_1, (512, 128), (1, 512), 0), out=buf270)
        del arg412_1
        buf271 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [add_130, mul_70, layer_outputs_93], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf271, buf270, arg413_1, arg415_1, arg414_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg413_1
        del arg414_1
        del arg415_1
        buf272 = reinterpret_tensor(buf269, (16384, 512), (512, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf271, (16384, 128), (128, 1), 0), reinterpret_tensor(arg416_1, (128, 512), (1, 128), 0), out=buf272)
        del arg416_1
        buf273 = reinterpret_tensor(buf272, (128, 128, 512), (65536, 512, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf273, arg417_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg417_1
        buf274 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf273, (16384, 512), (512, 1), 0), reinterpret_tensor(arg418_1, (512, 128), (1, 512), 0), out=buf274)
        del arg418_1
        buf275 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [add_132, mul_71, layer_outputs_95], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf275, buf274, arg419_1, arg421_1, arg420_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg419_1
        del arg420_1
        del arg421_1
        buf276 = reinterpret_tensor(buf273, (16384, 512), (512, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf275, (16384, 128), (128, 1), 0), reinterpret_tensor(arg386_1, (128, 512), (1, 128), 0), out=buf276)
        del arg386_1
        buf277 = reinterpret_tensor(buf276, (128, 128, 512), (65536, 512, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf277, arg387_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg387_1
        buf278 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf277, (16384, 512), (512, 1), 0), reinterpret_tensor(arg388_1, (512, 128), (1, 512), 0), out=buf278)
        del arg388_1
        buf279 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [add_134, mul_72, layer_output_17], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf279, buf278, arg389_1, arg391_1, arg390_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg389_1
        del arg390_1
        del arg391_1
        buf280 = reinterpret_tensor(buf277, (16384, 512), (512, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf279, (16384, 128), (128, 1), 0), reinterpret_tensor(arg392_1, (128, 512), (1, 128), 0), out=buf280)
        del arg392_1
        buf281 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [add_136, mul_73, layer_outputs_98], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf281, buf280, arg393_1, arg395_1, arg394_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg393_1
        del arg394_1
        del arg395_1
        buf282 = reinterpret_tensor(buf279, (16384, 128), (128, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (16384, 512), (512, 1), 0), reinterpret_tensor(arg446_1, (512, 128), (1, 512), 0), out=buf282)
        del arg446_1
        buf283 = reinterpret_tensor(buf282, (128, 128, 128), (16384, 128, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [mul_75, layer_input_39], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf283, arg447_1, arg449_1, arg448_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        buf284 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg423_1, reinterpret_tensor(buf283, (16384, 128), (128, 1), 0), reinterpret_tensor(arg422_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf284)
        del arg422_1
        del arg423_1
        buf285 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg425_1, reinterpret_tensor(buf283, (16384, 128), (128, 1), 0), reinterpret_tensor(arg424_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf285)
        del arg424_1
        del arg425_1
        buf286 = reinterpret_tensor(buf283, (16384, 128), (128, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg427_1, reinterpret_tensor(buf281, (16384, 512), (512, 1), 0), reinterpret_tensor(arg426_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf286)
        del arg426_1
        del arg427_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf287 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf284, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf285, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf286, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf284
        buf288 = buf287[0]
        del buf287
        buf292 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (16384, 128), (128, 1), 0), reinterpret_tensor(arg428_1, (128, 128), (1, 128), 0), out=buf292)
        del arg428_1
        buf293 = reinterpret_tensor(buf288, (16384, 128), (128, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (16384, 512), (512, 1), 0), reinterpret_tensor(arg442_1, (512, 128), (1, 512), 0), out=buf293)
        del arg442_1
        buf294 = reinterpret_tensor(buf292, (128, 128, 128), (16384, 128, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [mul_74, layer_input_37, add_141, mul_76, layer_outputs_100], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf294, arg429_1, buf293, arg443_1, arg445_1, arg444_1, arg431_1, arg430_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg429_1
        del arg430_1
        del arg431_1
        del arg443_1
        del arg444_1
        del arg445_1
        buf295 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf294, (16384, 128), (128, 1), 0), reinterpret_tensor(arg450_1, (128, 512), (1, 128), 0), out=buf295)
        del arg450_1
        buf296 = reinterpret_tensor(buf295, (128, 128, 512), (65536, 512, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf296, arg451_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg451_1
        buf297 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf296, (16384, 512), (512, 1), 0), reinterpret_tensor(arg452_1, (512, 128), (1, 512), 0), out=buf297)
        del arg452_1
        buf298 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [add_143, mul_77, layer_outputs_102], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf298, buf297, arg453_1, arg455_1, arg454_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg453_1
        del arg454_1
        del arg455_1
        buf299 = reinterpret_tensor(buf296, (16384, 512), (512, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf298, (16384, 128), (128, 1), 0), reinterpret_tensor(arg456_1, (128, 512), (1, 128), 0), out=buf299)
        del arg456_1
        buf300 = reinterpret_tensor(buf299, (128, 128, 512), (65536, 512, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf300, arg457_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg457_1
        buf301 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf300, (16384, 512), (512, 1), 0), reinterpret_tensor(arg458_1, (512, 128), (1, 512), 0), out=buf301)
        del arg458_1
        buf302 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [add_145, mul_78, layer_outputs_104], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf302, buf301, arg459_1, arg461_1, arg460_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg459_1
        del arg460_1
        del arg461_1
        buf303 = reinterpret_tensor(buf300, (16384, 512), (512, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (16384, 128), (128, 1), 0), reinterpret_tensor(arg462_1, (128, 512), (1, 128), 0), out=buf303)
        del arg462_1
        buf304 = reinterpret_tensor(buf303, (128, 128, 512), (65536, 512, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf304, arg463_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg463_1
        buf305 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf304, (16384, 512), (512, 1), 0), reinterpret_tensor(arg464_1, (512, 128), (1, 512), 0), out=buf305)
        del arg464_1
        buf306 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [add_147, mul_79, layer_outputs_106], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf306, buf305, arg465_1, arg467_1, arg466_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg465_1
        del arg466_1
        del arg467_1
        buf307 = reinterpret_tensor(buf304, (16384, 512), (512, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf306, (16384, 128), (128, 1), 0), reinterpret_tensor(arg432_1, (128, 512), (1, 128), 0), out=buf307)
        del arg432_1
        buf308 = reinterpret_tensor(buf307, (128, 128, 512), (65536, 512, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_79], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf308, arg433_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg433_1
        buf309 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (16384, 512), (512, 1), 0), reinterpret_tensor(arg434_1, (512, 128), (1, 512), 0), out=buf309)
        del arg434_1
        buf310 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [add_149, mul_80, layer_output_19], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf310, buf309, arg435_1, arg437_1, arg436_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg435_1
        del arg436_1
        del arg437_1
        buf311 = reinterpret_tensor(buf308, (16384, 512), (512, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (16384, 128), (128, 1), 0), reinterpret_tensor(arg438_1, (128, 512), (1, 128), 0), out=buf311)
        del arg438_1
        buf312 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [add_151, mul_81, layer_outputs_109], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf312, buf311, arg439_1, arg441_1, arg440_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg439_1
        del arg440_1
        del arg441_1
        buf313 = reinterpret_tensor(buf310, (16384, 128), (128, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (16384, 512), (512, 1), 0), reinterpret_tensor(arg492_1, (512, 128), (1, 512), 0), out=buf313)
        del arg492_1
        buf314 = reinterpret_tensor(buf313, (128, 128, 128), (16384, 128, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [mul_83, layer_input_43], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf314, arg493_1, arg495_1, arg494_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg493_1
        del arg494_1
        del arg495_1
        buf315 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg469_1, reinterpret_tensor(buf314, (16384, 128), (128, 1), 0), reinterpret_tensor(arg468_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf315)
        del arg468_1
        del arg469_1
        buf316 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg471_1, reinterpret_tensor(buf314, (16384, 128), (128, 1), 0), reinterpret_tensor(arg470_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf316)
        del arg470_1
        del arg471_1
        buf317 = reinterpret_tensor(buf314, (16384, 128), (128, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg473_1, reinterpret_tensor(buf312, (16384, 512), (512, 1), 0), reinterpret_tensor(arg472_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf317)
        del arg472_1
        del arg473_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf318 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf315, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf316, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf317, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf315
        buf319 = buf318[0]
        del buf318
        buf323 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf319, (16384, 128), (128, 1), 0), reinterpret_tensor(arg474_1, (128, 128), (1, 128), 0), out=buf323)
        del arg474_1
        buf324 = reinterpret_tensor(buf319, (16384, 128), (128, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (16384, 512), (512, 1), 0), reinterpret_tensor(arg488_1, (512, 128), (1, 512), 0), out=buf324)
        del arg488_1
        buf325 = reinterpret_tensor(buf323, (128, 128, 128), (16384, 128, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [mul_82, layer_input_41, add_156, mul_84, layer_outputs_111], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf325, arg475_1, buf324, arg489_1, arg491_1, arg490_1, arg477_1, arg476_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg475_1
        del arg476_1
        del arg477_1
        del arg489_1
        del arg490_1
        del arg491_1
        buf326 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf325, (16384, 128), (128, 1), 0), reinterpret_tensor(arg496_1, (128, 512), (1, 128), 0), out=buf326)
        del arg496_1
        buf327 = reinterpret_tensor(buf326, (128, 128, 512), (65536, 512, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf327, arg497_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg497_1
        buf328 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (16384, 512), (512, 1), 0), reinterpret_tensor(arg498_1, (512, 128), (1, 512), 0), out=buf328)
        del arg498_1
        buf329 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [add_158, mul_85, layer_outputs_113], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf329, buf328, arg499_1, arg501_1, arg500_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg499_1
        del arg500_1
        del arg501_1
        buf330 = reinterpret_tensor(buf327, (16384, 512), (512, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf329, (16384, 128), (128, 1), 0), reinterpret_tensor(arg502_1, (128, 512), (1, 128), 0), out=buf330)
        del arg502_1
        buf331 = reinterpret_tensor(buf330, (128, 128, 512), (65536, 512, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf331, arg503_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg503_1
        buf332 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf331, (16384, 512), (512, 1), 0), reinterpret_tensor(arg504_1, (512, 128), (1, 512), 0), out=buf332)
        del arg504_1
        buf333 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [add_160, mul_86, layer_outputs_115], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf333, buf332, arg505_1, arg507_1, arg506_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg505_1
        del arg506_1
        del arg507_1
        buf334 = reinterpret_tensor(buf331, (16384, 512), (512, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf333, (16384, 128), (128, 1), 0), reinterpret_tensor(arg508_1, (128, 512), (1, 128), 0), out=buf334)
        del arg508_1
        buf335 = reinterpret_tensor(buf334, (128, 128, 512), (65536, 512, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf335, arg509_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg509_1
        buf336 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (16384, 512), (512, 1), 0), reinterpret_tensor(arg510_1, (512, 128), (1, 512), 0), out=buf336)
        del arg510_1
        buf337 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [add_162, mul_87, layer_outputs_117], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf337, buf336, arg511_1, arg513_1, arg512_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg511_1
        del arg512_1
        del arg513_1
        buf338 = reinterpret_tensor(buf335, (16384, 512), (512, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (16384, 128), (128, 1), 0), reinterpret_tensor(arg478_1, (128, 512), (1, 128), 0), out=buf338)
        del arg478_1
        buf339 = reinterpret_tensor(buf338, (128, 128, 512), (65536, 512, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_87], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf339, arg479_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg479_1
        buf340 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf339, (16384, 512), (512, 1), 0), reinterpret_tensor(arg480_1, (512, 128), (1, 512), 0), out=buf340)
        del arg480_1
        buf341 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [add_164, mul_88, layer_output_21], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf341, buf340, arg481_1, arg483_1, arg482_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg481_1
        del arg482_1
        del arg483_1
        buf342 = reinterpret_tensor(buf339, (16384, 512), (512, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf341, (16384, 128), (128, 1), 0), reinterpret_tensor(arg484_1, (128, 512), (1, 128), 0), out=buf342)
        del arg484_1
        buf343 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [add_166, mul_89, layer_outputs_120], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf343, buf342, arg485_1, arg487_1, arg486_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg485_1
        del arg486_1
        del arg487_1
        buf344 = reinterpret_tensor(buf341, (16384, 128), (128, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (16384, 512), (512, 1), 0), reinterpret_tensor(arg538_1, (512, 128), (1, 512), 0), out=buf344)
        del arg538_1
        buf345 = reinterpret_tensor(buf344, (128, 128, 128), (16384, 128, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [mul_91, layer_input_47], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf345, arg539_1, arg541_1, arg540_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg539_1
        del arg540_1
        del arg541_1
        buf346 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg515_1, reinterpret_tensor(buf345, (16384, 128), (128, 1), 0), reinterpret_tensor(arg514_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf346)
        del arg514_1
        del arg515_1
        buf347 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg517_1, reinterpret_tensor(buf345, (16384, 128), (128, 1), 0), reinterpret_tensor(arg516_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf347)
        del arg516_1
        del arg517_1
        buf348 = reinterpret_tensor(buf345, (16384, 128), (128, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg519_1, reinterpret_tensor(buf343, (16384, 512), (512, 1), 0), reinterpret_tensor(arg518_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf348)
        del arg518_1
        del arg519_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf349 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf346, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf347, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf348, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf346
        buf350 = buf349[0]
        del buf349
        buf354 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf350, (16384, 128), (128, 1), 0), reinterpret_tensor(arg520_1, (128, 128), (1, 128), 0), out=buf354)
        del arg520_1
        buf355 = reinterpret_tensor(buf350, (16384, 128), (128, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf343, (16384, 512), (512, 1), 0), reinterpret_tensor(arg534_1, (512, 128), (1, 512), 0), out=buf355)
        del arg534_1
        buf356 = reinterpret_tensor(buf354, (128, 128, 128), (16384, 128, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [mul_90, layer_input_45, add_171, mul_92, layer_outputs_122], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf356, arg521_1, buf355, arg535_1, arg537_1, arg536_1, arg523_1, arg522_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg521_1
        del arg522_1
        del arg523_1
        del arg535_1
        del arg536_1
        del arg537_1
        buf357 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (16384, 128), (128, 1), 0), reinterpret_tensor(arg542_1, (128, 512), (1, 128), 0), out=buf357)
        del arg542_1
        buf358 = reinterpret_tensor(buf357, (128, 128, 512), (65536, 512, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_89], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf358, arg543_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg543_1
        buf359 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf358, (16384, 512), (512, 1), 0), reinterpret_tensor(arg544_1, (512, 128), (1, 512), 0), out=buf359)
        del arg544_1
        buf360 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [add_173, mul_93, layer_outputs_124], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf360, buf359, arg545_1, arg547_1, arg546_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg545_1
        del arg546_1
        del arg547_1
        buf361 = reinterpret_tensor(buf358, (16384, 512), (512, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf360, (16384, 128), (128, 1), 0), reinterpret_tensor(arg548_1, (128, 512), (1, 128), 0), out=buf361)
        del arg548_1
        buf362 = reinterpret_tensor(buf361, (128, 128, 512), (65536, 512, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf362, arg549_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg549_1
        buf363 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (16384, 512), (512, 1), 0), reinterpret_tensor(arg550_1, (512, 128), (1, 512), 0), out=buf363)
        del arg550_1
        buf364 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [add_175, mul_94, layer_outputs_126], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf364, buf363, arg551_1, arg553_1, arg552_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg551_1
        del arg552_1
        del arg553_1
        buf365 = reinterpret_tensor(buf362, (16384, 512), (512, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (16384, 128), (128, 1), 0), reinterpret_tensor(arg554_1, (128, 512), (1, 128), 0), out=buf365)
        del arg554_1
        buf366 = reinterpret_tensor(buf365, (128, 128, 512), (65536, 512, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf366, arg555_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg555_1
        buf367 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf366, (16384, 512), (512, 1), 0), reinterpret_tensor(arg556_1, (512, 128), (1, 512), 0), out=buf367)
        del arg556_1
        buf368 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [add_177, mul_95, layer_outputs_128], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf368, buf367, arg557_1, arg559_1, arg558_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg557_1
        del arg558_1
        del arg559_1
        buf369 = reinterpret_tensor(buf366, (16384, 512), (512, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf368, (16384, 128), (128, 1), 0), reinterpret_tensor(arg524_1, (128, 512), (1, 128), 0), out=buf369)
        del arg524_1
        buf370 = reinterpret_tensor(buf369, (128, 128, 512), (65536, 512, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_95], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf370, arg525_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg525_1
        buf371 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (16384, 512), (512, 1), 0), reinterpret_tensor(arg526_1, (512, 128), (1, 512), 0), out=buf371)
        del arg526_1
        buf372 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [add_179, mul_96, layer_output_23], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf372, buf371, arg527_1, arg529_1, arg528_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg527_1
        del arg528_1
        del arg529_1
        buf373 = reinterpret_tensor(buf370, (16384, 512), (512, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf372, (16384, 128), (128, 1), 0), reinterpret_tensor(arg530_1, (128, 512), (1, 128), 0), out=buf373)
        del arg530_1
        buf374 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [add_181, mul_97, layer_outputs_131], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf374, buf373, arg531_1, arg533_1, arg532_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg531_1
        del arg532_1
        del arg533_1
        buf375 = reinterpret_tensor(buf372, (16384, 128), (128, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (16384, 512), (512, 1), 0), reinterpret_tensor(arg584_1, (512, 128), (1, 512), 0), out=buf375)
        del arg584_1
        buf376 = reinterpret_tensor(buf375, (128, 128, 128), (16384, 128, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [mul_99, layer_input_51], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf376, arg585_1, arg587_1, arg586_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg585_1
        del arg586_1
        del arg587_1
        buf377 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg561_1, reinterpret_tensor(buf376, (16384, 128), (128, 1), 0), reinterpret_tensor(arg560_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf377)
        del arg560_1
        del arg561_1
        buf378 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg563_1, reinterpret_tensor(buf376, (16384, 128), (128, 1), 0), reinterpret_tensor(arg562_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf378)
        del arg562_1
        del arg563_1
        buf379 = reinterpret_tensor(buf376, (16384, 128), (128, 1), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg565_1, reinterpret_tensor(buf374, (16384, 512), (512, 1), 0), reinterpret_tensor(arg564_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf379)
        del arg564_1
        del arg565_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf380 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf377, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf378, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf379, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf377
        buf381 = buf380[0]
        del buf380
        buf385 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (16384, 128), (128, 1), 0), reinterpret_tensor(arg566_1, (128, 128), (1, 128), 0), out=buf385)
        del arg566_1
        buf386 = reinterpret_tensor(buf381, (16384, 128), (128, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (16384, 512), (512, 1), 0), reinterpret_tensor(arg580_1, (512, 128), (1, 512), 0), out=buf386)
        del arg580_1
        buf387 = reinterpret_tensor(buf385, (128, 128, 128), (16384, 128, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [mul_98, layer_input_49, add_186, mul_100, layer_outputs_133], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf387, arg567_1, buf386, arg581_1, arg583_1, arg582_1, arg569_1, arg568_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg567_1
        del arg568_1
        del arg569_1
        del arg581_1
        del arg582_1
        del arg583_1
        buf388 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (16384, 128), (128, 1), 0), reinterpret_tensor(arg588_1, (128, 512), (1, 128), 0), out=buf388)
        del arg588_1
        buf389 = reinterpret_tensor(buf388, (128, 128, 512), (65536, 512, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_97], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf389, arg589_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg589_1
        buf390 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (16384, 512), (512, 1), 0), reinterpret_tensor(arg590_1, (512, 128), (1, 512), 0), out=buf390)
        del arg590_1
        buf391 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [add_188, mul_101, layer_outputs_135], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf391, buf390, arg591_1, arg593_1, arg592_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg591_1
        del arg592_1
        del arg593_1
        buf392 = reinterpret_tensor(buf389, (16384, 512), (512, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf391, (16384, 128), (128, 1), 0), reinterpret_tensor(arg594_1, (128, 512), (1, 128), 0), out=buf392)
        del arg594_1
        buf393 = reinterpret_tensor(buf392, (128, 128, 512), (65536, 512, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_99], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf393, arg595_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg595_1
        buf394 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf393, (16384, 512), (512, 1), 0), reinterpret_tensor(arg596_1, (512, 128), (1, 512), 0), out=buf394)
        del arg596_1
        buf395 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [add_190, mul_102, layer_outputs_137], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf395, buf394, arg597_1, arg599_1, arg598_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg597_1
        del arg598_1
        del arg599_1
        buf396 = reinterpret_tensor(buf393, (16384, 512), (512, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf395, (16384, 128), (128, 1), 0), reinterpret_tensor(arg600_1, (128, 512), (1, 128), 0), out=buf396)
        del arg600_1
        buf397 = reinterpret_tensor(buf396, (128, 128, 512), (65536, 512, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf397, arg601_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg601_1
        buf398 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf397, (16384, 512), (512, 1), 0), reinterpret_tensor(arg602_1, (512, 128), (1, 512), 0), out=buf398)
        del arg602_1
        buf399 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [add_192, mul_103, layer_outputs_139], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf399, buf398, arg603_1, arg605_1, arg604_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg603_1
        del arg604_1
        del arg605_1
        buf400 = reinterpret_tensor(buf397, (16384, 512), (512, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (16384, 128), (128, 1), 0), reinterpret_tensor(arg570_1, (128, 512), (1, 128), 0), out=buf400)
        del arg570_1
        buf401 = reinterpret_tensor(buf400, (128, 128, 512), (65536, 512, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_103], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf401, arg571_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg571_1
        buf402 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf401, (16384, 512), (512, 1), 0), reinterpret_tensor(arg572_1, (512, 128), (1, 512), 0), out=buf402)
        del arg572_1
        buf403 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [add_194, mul_104, layer_output_25], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf403, buf402, arg573_1, arg575_1, arg574_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg573_1
        del arg574_1
        del arg575_1
        buf404 = reinterpret_tensor(buf401, (16384, 512), (512, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf403, (16384, 128), (128, 1), 0), reinterpret_tensor(arg576_1, (128, 512), (1, 128), 0), out=buf404)
        del arg576_1
        buf405 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [add_196, mul_105, layer_outputs_142], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf405, buf404, arg577_1, arg579_1, arg578_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg577_1
        del arg578_1
        del arg579_1
        buf406 = reinterpret_tensor(buf403, (16384, 128), (128, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf405, (16384, 512), (512, 1), 0), reinterpret_tensor(arg630_1, (512, 128), (1, 512), 0), out=buf406)
        del arg630_1
        buf407 = reinterpret_tensor(buf406, (128, 128, 128), (16384, 128, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [mul_107, layer_input_55], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf407, arg631_1, arg633_1, arg632_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg631_1
        del arg632_1
        del arg633_1
        buf408 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg607_1, reinterpret_tensor(buf407, (16384, 128), (128, 1), 0), reinterpret_tensor(arg606_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf408)
        del arg606_1
        del arg607_1
        buf409 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg609_1, reinterpret_tensor(buf407, (16384, 128), (128, 1), 0), reinterpret_tensor(arg608_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf409)
        del arg608_1
        del arg609_1
        buf410 = reinterpret_tensor(buf407, (16384, 128), (128, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg611_1, reinterpret_tensor(buf405, (16384, 512), (512, 1), 0), reinterpret_tensor(arg610_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf410)
        del arg610_1
        del arg611_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf411 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf408, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf409, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf410, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf408
        buf412 = buf411[0]
        del buf411
        buf416 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf412, (16384, 128), (128, 1), 0), reinterpret_tensor(arg612_1, (128, 128), (1, 128), 0), out=buf416)
        del arg612_1
        buf417 = reinterpret_tensor(buf412, (16384, 128), (128, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf405, (16384, 512), (512, 1), 0), reinterpret_tensor(arg626_1, (512, 128), (1, 512), 0), out=buf417)
        del arg626_1
        buf418 = reinterpret_tensor(buf416, (128, 128, 128), (16384, 128, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [mul_106, layer_input_53, add_201, mul_108, layer_outputs_144], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf418, arg613_1, buf417, arg627_1, arg629_1, arg628_1, arg615_1, arg614_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg613_1
        del arg614_1
        del arg615_1
        del arg627_1
        del arg628_1
        del arg629_1
        buf419 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf418, (16384, 128), (128, 1), 0), reinterpret_tensor(arg634_1, (128, 512), (1, 128), 0), out=buf419)
        del arg634_1
        buf420 = reinterpret_tensor(buf419, (128, 128, 512), (65536, 512, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_105], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf420, arg635_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg635_1
        buf421 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf420, (16384, 512), (512, 1), 0), reinterpret_tensor(arg636_1, (512, 128), (1, 512), 0), out=buf421)
        del arg636_1
        buf422 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [add_203, mul_109, layer_outputs_146], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf422, buf421, arg637_1, arg639_1, arg638_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg637_1
        del arg638_1
        del arg639_1
        buf423 = reinterpret_tensor(buf420, (16384, 512), (512, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf422, (16384, 128), (128, 1), 0), reinterpret_tensor(arg640_1, (128, 512), (1, 128), 0), out=buf423)
        del arg640_1
        buf424 = reinterpret_tensor(buf423, (128, 128, 512), (65536, 512, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf424, arg641_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg641_1
        buf425 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf424, (16384, 512), (512, 1), 0), reinterpret_tensor(arg642_1, (512, 128), (1, 512), 0), out=buf425)
        del arg642_1
        buf426 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [add_205, mul_110, layer_outputs_148], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf426, buf425, arg643_1, arg645_1, arg644_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg643_1
        del arg644_1
        del arg645_1
        buf427 = reinterpret_tensor(buf424, (16384, 512), (512, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf426, (16384, 128), (128, 1), 0), reinterpret_tensor(arg646_1, (128, 512), (1, 128), 0), out=buf427)
        del arg646_1
        buf428 = reinterpret_tensor(buf427, (128, 128, 512), (65536, 512, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_109], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf428, arg647_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg647_1
        buf429 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf428, (16384, 512), (512, 1), 0), reinterpret_tensor(arg648_1, (512, 128), (1, 512), 0), out=buf429)
        del arg648_1
        buf430 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [add_207, mul_111, layer_outputs_150], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf430, buf429, arg649_1, arg651_1, arg650_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg649_1
        del arg650_1
        del arg651_1
        buf431 = reinterpret_tensor(buf428, (16384, 512), (512, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf430, (16384, 128), (128, 1), 0), reinterpret_tensor(arg616_1, (128, 512), (1, 128), 0), out=buf431)
        del arg616_1
        buf432 = reinterpret_tensor(buf431, (128, 128, 512), (65536, 512, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf432, arg617_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg617_1
        buf433 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf432, (16384, 512), (512, 1), 0), reinterpret_tensor(arg618_1, (512, 128), (1, 512), 0), out=buf433)
        del arg618_1
        buf434 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [add_209, mul_112, layer_output_27], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf434, buf433, arg619_1, arg621_1, arg620_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg619_1
        del arg620_1
        del arg621_1
        buf435 = reinterpret_tensor(buf432, (16384, 512), (512, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf434, (16384, 128), (128, 1), 0), reinterpret_tensor(arg622_1, (128, 512), (1, 128), 0), out=buf435)
        del arg622_1
        buf436 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [add_211, mul_113, layer_outputs_153], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf436, buf435, arg623_1, arg625_1, arg624_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg623_1
        del arg624_1
        del arg625_1
        buf437 = reinterpret_tensor(buf434, (16384, 128), (128, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (16384, 512), (512, 1), 0), reinterpret_tensor(arg676_1, (512, 128), (1, 512), 0), out=buf437)
        del arg676_1
        buf438 = reinterpret_tensor(buf437, (128, 128, 128), (16384, 128, 1), 0); del buf437  # reuse
        # Topologically Sorted Source Nodes: [mul_115, layer_input_59], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf438, arg677_1, arg679_1, arg678_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg677_1
        del arg678_1
        del arg679_1
        buf439 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg653_1, reinterpret_tensor(buf438, (16384, 128), (128, 1), 0), reinterpret_tensor(arg652_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf439)
        del arg652_1
        del arg653_1
        buf440 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg655_1, reinterpret_tensor(buf438, (16384, 128), (128, 1), 0), reinterpret_tensor(arg654_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf440)
        del arg654_1
        del arg655_1
        buf441 = reinterpret_tensor(buf438, (16384, 128), (128, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg657_1, reinterpret_tensor(buf436, (16384, 512), (512, 1), 0), reinterpret_tensor(arg656_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf441)
        del arg656_1
        del arg657_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf442 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf439, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf440, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf441, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf439
        buf443 = buf442[0]
        del buf442
        buf447 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf443, (16384, 128), (128, 1), 0), reinterpret_tensor(arg658_1, (128, 128), (1, 128), 0), out=buf447)
        del arg658_1
        buf448 = reinterpret_tensor(buf443, (16384, 128), (128, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (16384, 512), (512, 1), 0), reinterpret_tensor(arg672_1, (512, 128), (1, 512), 0), out=buf448)
        del arg672_1
        buf449 = reinterpret_tensor(buf447, (128, 128, 128), (16384, 128, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [mul_114, layer_input_57, add_216, mul_116, layer_outputs_155], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf449, arg659_1, buf448, arg673_1, arg675_1, arg674_1, arg661_1, arg660_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg659_1
        del arg660_1
        del arg661_1
        del arg673_1
        del arg674_1
        del arg675_1
        buf450 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf449, (16384, 128), (128, 1), 0), reinterpret_tensor(arg680_1, (128, 512), (1, 128), 0), out=buf450)
        del arg680_1
        buf451 = reinterpret_tensor(buf450, (128, 128, 512), (65536, 512, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_113], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf451, arg681_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg681_1
        buf452 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf451, (16384, 512), (512, 1), 0), reinterpret_tensor(arg682_1, (512, 128), (1, 512), 0), out=buf452)
        del arg682_1
        buf453 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [add_218, mul_117, layer_outputs_157], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf453, buf452, arg683_1, arg685_1, arg684_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg683_1
        del arg684_1
        del arg685_1
        buf454 = reinterpret_tensor(buf451, (16384, 512), (512, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf453, (16384, 128), (128, 1), 0), reinterpret_tensor(arg686_1, (128, 512), (1, 128), 0), out=buf454)
        del arg686_1
        buf455 = reinterpret_tensor(buf454, (128, 128, 512), (65536, 512, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf455, arg687_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg687_1
        buf456 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf455, (16384, 512), (512, 1), 0), reinterpret_tensor(arg688_1, (512, 128), (1, 512), 0), out=buf456)
        del arg688_1
        buf457 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [add_220, mul_118, layer_outputs_159], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf457, buf456, arg689_1, arg691_1, arg690_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg689_1
        del arg690_1
        del arg691_1
        buf458 = reinterpret_tensor(buf455, (16384, 512), (512, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf457, (16384, 128), (128, 1), 0), reinterpret_tensor(arg692_1, (128, 512), (1, 128), 0), out=buf458)
        del arg692_1
        buf459 = reinterpret_tensor(buf458, (128, 128, 512), (65536, 512, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_117], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf459, arg693_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg693_1
        buf460 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf459, (16384, 512), (512, 1), 0), reinterpret_tensor(arg694_1, (512, 128), (1, 512), 0), out=buf460)
        del arg694_1
        buf461 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [add_222, mul_119, layer_outputs_161], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf461, buf460, arg695_1, arg697_1, arg696_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg695_1
        del arg696_1
        del arg697_1
        buf462 = reinterpret_tensor(buf459, (16384, 512), (512, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf461, (16384, 128), (128, 1), 0), reinterpret_tensor(arg662_1, (128, 512), (1, 128), 0), out=buf462)
        del arg662_1
        buf463 = reinterpret_tensor(buf462, (128, 128, 512), (65536, 512, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_119], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf463, arg663_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg663_1
        buf464 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf463, (16384, 512), (512, 1), 0), reinterpret_tensor(arg664_1, (512, 128), (1, 512), 0), out=buf464)
        del arg664_1
        buf465 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [add_224, mul_120, layer_output_29], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf465, buf464, arg665_1, arg667_1, arg666_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg665_1
        del arg666_1
        del arg667_1
        buf466 = reinterpret_tensor(buf463, (16384, 512), (512, 1), 0); del buf463  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf465, (16384, 128), (128, 1), 0), reinterpret_tensor(arg668_1, (128, 512), (1, 128), 0), out=buf466)
        del arg668_1
        buf467 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [add_226, mul_121, layer_outputs_164], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf467, buf466, arg669_1, arg671_1, arg670_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg669_1
        del arg670_1
        del arg671_1
        buf468 = reinterpret_tensor(buf465, (16384, 128), (128, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (16384, 512), (512, 1), 0), reinterpret_tensor(arg722_1, (512, 128), (1, 512), 0), out=buf468)
        del arg722_1
        buf469 = reinterpret_tensor(buf468, (128, 128, 128), (16384, 128, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [mul_123, layer_input_63], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf469, arg723_1, arg725_1, arg724_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg723_1
        del arg724_1
        del arg725_1
        buf470 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg699_1, reinterpret_tensor(buf469, (16384, 128), (128, 1), 0), reinterpret_tensor(arg698_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf470)
        del arg698_1
        del arg699_1
        buf471 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg701_1, reinterpret_tensor(buf469, (16384, 128), (128, 1), 0), reinterpret_tensor(arg700_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf471)
        del arg700_1
        del arg701_1
        buf472 = reinterpret_tensor(buf469, (16384, 128), (128, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg703_1, reinterpret_tensor(buf467, (16384, 512), (512, 1), 0), reinterpret_tensor(arg702_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf472)
        del arg702_1
        del arg703_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf473 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf470, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf471, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf472, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf470
        buf474 = buf473[0]
        del buf473
        buf478 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf474, (16384, 128), (128, 1), 0), reinterpret_tensor(arg704_1, (128, 128), (1, 128), 0), out=buf478)
        del arg704_1
        buf479 = reinterpret_tensor(buf474, (16384, 128), (128, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf467, (16384, 512), (512, 1), 0), reinterpret_tensor(arg718_1, (512, 128), (1, 512), 0), out=buf479)
        del arg718_1
        buf480 = reinterpret_tensor(buf478, (128, 128, 128), (16384, 128, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [mul_122, layer_input_61, add_231, mul_124, layer_outputs_166], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf480, arg705_1, buf479, arg719_1, arg721_1, arg720_1, arg707_1, arg706_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg705_1
        del arg706_1
        del arg707_1
        del arg719_1
        del arg720_1
        del arg721_1
        buf481 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf480, (16384, 128), (128, 1), 0), reinterpret_tensor(arg726_1, (128, 512), (1, 128), 0), out=buf481)
        del arg726_1
        buf482 = reinterpret_tensor(buf481, (128, 128, 512), (65536, 512, 1), 0); del buf481  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_121], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf482, arg727_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg727_1
        buf483 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf482, (16384, 512), (512, 1), 0), reinterpret_tensor(arg728_1, (512, 128), (1, 512), 0), out=buf483)
        del arg728_1
        buf484 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [add_233, mul_125, layer_outputs_168], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf484, buf483, arg729_1, arg731_1, arg730_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg729_1
        del arg730_1
        del arg731_1
        buf485 = reinterpret_tensor(buf482, (16384, 512), (512, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf484, (16384, 128), (128, 1), 0), reinterpret_tensor(arg732_1, (128, 512), (1, 128), 0), out=buf485)
        del arg732_1
        buf486 = reinterpret_tensor(buf485, (128, 128, 512), (65536, 512, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_123], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf486, arg733_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg733_1
        buf487 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf486, (16384, 512), (512, 1), 0), reinterpret_tensor(arg734_1, (512, 128), (1, 512), 0), out=buf487)
        del arg734_1
        buf488 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [add_235, mul_126, layer_outputs_170], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf488, buf487, arg735_1, arg737_1, arg736_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg735_1
        del arg736_1
        del arg737_1
        buf489 = reinterpret_tensor(buf486, (16384, 512), (512, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf488, (16384, 128), (128, 1), 0), reinterpret_tensor(arg738_1, (128, 512), (1, 128), 0), out=buf489)
        del arg738_1
        buf490 = reinterpret_tensor(buf489, (128, 128, 512), (65536, 512, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_125], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf490, arg739_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg739_1
        buf491 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf490, (16384, 512), (512, 1), 0), reinterpret_tensor(arg740_1, (512, 128), (1, 512), 0), out=buf491)
        del arg740_1
        buf492 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [add_237, mul_127, layer_outputs_172], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf492, buf491, arg741_1, arg743_1, arg742_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg741_1
        del arg742_1
        del arg743_1
        buf493 = reinterpret_tensor(buf490, (16384, 512), (512, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (16384, 128), (128, 1), 0), reinterpret_tensor(arg708_1, (128, 512), (1, 128), 0), out=buf493)
        del arg708_1
        buf494 = reinterpret_tensor(buf493, (128, 128, 512), (65536, 512, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_127], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf494, arg709_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg709_1
        buf495 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (16384, 512), (512, 1), 0), reinterpret_tensor(arg710_1, (512, 128), (1, 512), 0), out=buf495)
        del arg710_1
        buf496 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [add_239, mul_128, layer_output_31], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf496, buf495, arg711_1, arg713_1, arg712_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg711_1
        del arg712_1
        del arg713_1
        buf497 = reinterpret_tensor(buf494, (16384, 512), (512, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf496, (16384, 128), (128, 1), 0), reinterpret_tensor(arg714_1, (128, 512), (1, 128), 0), out=buf497)
        del arg714_1
        buf498 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [add_241, mul_129, layer_outputs_175], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf498, buf497, arg715_1, arg717_1, arg716_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg715_1
        del arg716_1
        del arg717_1
        buf499 = reinterpret_tensor(buf496, (16384, 128), (128, 1), 0); del buf496  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (16384, 512), (512, 1), 0), reinterpret_tensor(arg768_1, (512, 128), (1, 512), 0), out=buf499)
        del arg768_1
        buf500 = reinterpret_tensor(buf499, (128, 128, 128), (16384, 128, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [mul_131, layer_input_67], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf500, arg769_1, arg771_1, arg770_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg769_1
        del arg770_1
        del arg771_1
        buf501 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg745_1, reinterpret_tensor(buf500, (16384, 128), (128, 1), 0), reinterpret_tensor(arg744_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf501)
        del arg744_1
        del arg745_1
        buf502 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg747_1, reinterpret_tensor(buf500, (16384, 128), (128, 1), 0), reinterpret_tensor(arg746_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf502)
        del arg746_1
        del arg747_1
        buf503 = reinterpret_tensor(buf500, (16384, 128), (128, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg749_1, reinterpret_tensor(buf498, (16384, 512), (512, 1), 0), reinterpret_tensor(arg748_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf503)
        del arg748_1
        del arg749_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf504 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf501, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf502, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf503, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf501
        buf505 = buf504[0]
        del buf504
        buf509 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (16384, 128), (128, 1), 0), reinterpret_tensor(arg750_1, (128, 128), (1, 128), 0), out=buf509)
        del arg750_1
        buf510 = reinterpret_tensor(buf505, (16384, 128), (128, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf498, (16384, 512), (512, 1), 0), reinterpret_tensor(arg764_1, (512, 128), (1, 512), 0), out=buf510)
        del arg764_1
        buf511 = reinterpret_tensor(buf509, (128, 128, 128), (16384, 128, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [mul_130, layer_input_65, add_246, mul_132, layer_outputs_177], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf511, arg751_1, buf510, arg765_1, arg767_1, arg766_1, arg753_1, arg752_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg751_1
        del arg752_1
        del arg753_1
        del arg765_1
        del arg766_1
        del arg767_1
        buf512 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf511, (16384, 128), (128, 1), 0), reinterpret_tensor(arg772_1, (128, 512), (1, 128), 0), out=buf512)
        del arg772_1
        buf513 = reinterpret_tensor(buf512, (128, 128, 512), (65536, 512, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf513, arg773_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg773_1
        buf514 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf513, (16384, 512), (512, 1), 0), reinterpret_tensor(arg774_1, (512, 128), (1, 512), 0), out=buf514)
        del arg774_1
        buf515 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [add_248, mul_133, layer_outputs_179], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf515, buf514, arg775_1, arg777_1, arg776_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg775_1
        del arg776_1
        del arg777_1
        buf516 = reinterpret_tensor(buf513, (16384, 512), (512, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf515, (16384, 128), (128, 1), 0), reinterpret_tensor(arg778_1, (128, 512), (1, 128), 0), out=buf516)
        del arg778_1
        buf517 = reinterpret_tensor(buf516, (128, 128, 512), (65536, 512, 1), 0); del buf516  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_131], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf517, arg779_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg779_1
        buf518 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf517, (16384, 512), (512, 1), 0), reinterpret_tensor(arg780_1, (512, 128), (1, 512), 0), out=buf518)
        del arg780_1
        buf519 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [add_250, mul_134, layer_outputs_181], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf519, buf518, arg781_1, arg783_1, arg782_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg781_1
        del arg782_1
        del arg783_1
        buf520 = reinterpret_tensor(buf517, (16384, 512), (512, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf519, (16384, 128), (128, 1), 0), reinterpret_tensor(arg784_1, (128, 512), (1, 128), 0), out=buf520)
        del arg784_1
        buf521 = reinterpret_tensor(buf520, (128, 128, 512), (65536, 512, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_133], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf521, arg785_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg785_1
        buf522 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf521, (16384, 512), (512, 1), 0), reinterpret_tensor(arg786_1, (512, 128), (1, 512), 0), out=buf522)
        del arg786_1
        buf523 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [add_252, mul_135, layer_outputs_183], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf523, buf522, arg787_1, arg789_1, arg788_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg787_1
        del arg788_1
        del arg789_1
        buf524 = reinterpret_tensor(buf521, (16384, 512), (512, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf523, (16384, 128), (128, 1), 0), reinterpret_tensor(arg754_1, (128, 512), (1, 128), 0), out=buf524)
        del arg754_1
        buf525 = reinterpret_tensor(buf524, (128, 128, 512), (65536, 512, 1), 0); del buf524  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_135], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf525, arg755_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg755_1
        buf526 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf525, (16384, 512), (512, 1), 0), reinterpret_tensor(arg756_1, (512, 128), (1, 512), 0), out=buf526)
        del arg756_1
        buf527 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [add_254, mul_136, layer_output_33], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf527, buf526, arg757_1, arg759_1, arg758_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg757_1
        del arg758_1
        del arg759_1
        buf528 = reinterpret_tensor(buf525, (16384, 512), (512, 1), 0); del buf525  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf527, (16384, 128), (128, 1), 0), reinterpret_tensor(arg760_1, (128, 512), (1, 128), 0), out=buf528)
        del arg760_1
        buf529 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [add_256, mul_137, layer_outputs_186], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf529, buf528, arg761_1, arg763_1, arg762_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg761_1
        del arg762_1
        del arg763_1
        buf530 = reinterpret_tensor(buf527, (16384, 128), (128, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (16384, 512), (512, 1), 0), reinterpret_tensor(arg814_1, (512, 128), (1, 512), 0), out=buf530)
        del arg814_1
        buf531 = reinterpret_tensor(buf530, (128, 128, 128), (16384, 128, 1), 0); del buf530  # reuse
        # Topologically Sorted Source Nodes: [mul_139, layer_input_71], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf531, arg815_1, arg817_1, arg816_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg815_1
        del arg816_1
        del arg817_1
        buf532 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg791_1, reinterpret_tensor(buf531, (16384, 128), (128, 1), 0), reinterpret_tensor(arg790_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf532)
        del arg790_1
        del arg791_1
        buf533 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg793_1, reinterpret_tensor(buf531, (16384, 128), (128, 1), 0), reinterpret_tensor(arg792_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf533)
        del arg792_1
        del arg793_1
        buf534 = reinterpret_tensor(buf531, (16384, 128), (128, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg795_1, reinterpret_tensor(buf529, (16384, 512), (512, 1), 0), reinterpret_tensor(arg794_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf534)
        del arg794_1
        del arg795_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf535 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf532, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf533, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf534, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf532
        buf536 = buf535[0]
        del buf535
        buf540 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf536, (16384, 128), (128, 1), 0), reinterpret_tensor(arg796_1, (128, 128), (1, 128), 0), out=buf540)
        del arg796_1
        buf541 = reinterpret_tensor(buf536, (16384, 128), (128, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (16384, 512), (512, 1), 0), reinterpret_tensor(arg810_1, (512, 128), (1, 512), 0), out=buf541)
        del arg810_1
        buf542 = reinterpret_tensor(buf540, (128, 128, 128), (16384, 128, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [mul_138, layer_input_69, add_261, mul_140, layer_outputs_188], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf542, arg797_1, buf541, arg811_1, arg813_1, arg812_1, arg799_1, arg798_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg797_1
        del arg798_1
        del arg799_1
        del arg811_1
        del arg812_1
        del arg813_1
        buf543 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf542, (16384, 128), (128, 1), 0), reinterpret_tensor(arg818_1, (128, 512), (1, 128), 0), out=buf543)
        del arg818_1
        buf544 = reinterpret_tensor(buf543, (128, 128, 512), (65536, 512, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_137], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf544, arg819_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg819_1
        buf545 = buf541; del buf541  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf544, (16384, 512), (512, 1), 0), reinterpret_tensor(arg820_1, (512, 128), (1, 512), 0), out=buf545)
        del arg820_1
        buf546 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [add_263, mul_141, layer_outputs_190], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf546, buf545, arg821_1, arg823_1, arg822_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg821_1
        del arg822_1
        del arg823_1
        buf547 = reinterpret_tensor(buf544, (16384, 512), (512, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (16384, 128), (128, 1), 0), reinterpret_tensor(arg824_1, (128, 512), (1, 128), 0), out=buf547)
        del arg824_1
        buf548 = reinterpret_tensor(buf547, (128, 128, 512), (65536, 512, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_139], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf548, arg825_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg825_1
        buf549 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (16384, 512), (512, 1), 0), reinterpret_tensor(arg826_1, (512, 128), (1, 512), 0), out=buf549)
        del arg826_1
        buf550 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [add_265, mul_142, layer_outputs_192], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf550, buf549, arg827_1, arg829_1, arg828_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg827_1
        del arg828_1
        del arg829_1
        buf551 = reinterpret_tensor(buf548, (16384, 512), (512, 1), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf550, (16384, 128), (128, 1), 0), reinterpret_tensor(arg830_1, (128, 512), (1, 128), 0), out=buf551)
        del arg830_1
        buf552 = reinterpret_tensor(buf551, (128, 128, 512), (65536, 512, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_141], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf552, arg831_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg831_1
        buf553 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf552, (16384, 512), (512, 1), 0), reinterpret_tensor(arg832_1, (512, 128), (1, 512), 0), out=buf553)
        del arg832_1
        buf554 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [add_267, mul_143, layer_outputs_194], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf554, buf553, arg833_1, arg835_1, arg834_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg833_1
        del arg834_1
        del arg835_1
        buf555 = reinterpret_tensor(buf552, (16384, 512), (512, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf554, (16384, 128), (128, 1), 0), reinterpret_tensor(arg800_1, (128, 512), (1, 128), 0), out=buf555)
        del arg800_1
        buf556 = reinterpret_tensor(buf555, (128, 128, 512), (65536, 512, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_143], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf556, arg801_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg801_1
        buf557 = buf553; del buf553  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf556, (16384, 512), (512, 1), 0), reinterpret_tensor(arg802_1, (512, 128), (1, 512), 0), out=buf557)
        del arg802_1
        buf558 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [add_269, mul_144, layer_output_35], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf558, buf557, arg803_1, arg805_1, arg804_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg803_1
        del arg804_1
        del arg805_1
        buf559 = reinterpret_tensor(buf556, (16384, 512), (512, 1), 0); del buf556  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf558, (16384, 128), (128, 1), 0), reinterpret_tensor(arg806_1, (128, 512), (1, 128), 0), out=buf559)
        del arg806_1
        buf560 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [add_271, mul_145, layer_outputs_197], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf560, buf559, arg807_1, arg809_1, arg808_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg807_1
        del arg808_1
        del arg809_1
        buf561 = reinterpret_tensor(buf558, (16384, 128), (128, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf560, (16384, 512), (512, 1), 0), reinterpret_tensor(arg860_1, (512, 128), (1, 512), 0), out=buf561)
        del arg860_1
        buf562 = reinterpret_tensor(buf561, (128, 128, 128), (16384, 128, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [mul_147, layer_input_75], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf562, arg861_1, arg863_1, arg862_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg861_1
        del arg862_1
        del arg863_1
        buf563 = buf557; del buf557  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg837_1, reinterpret_tensor(buf562, (16384, 128), (128, 1), 0), reinterpret_tensor(arg836_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf563)
        del arg836_1
        del arg837_1
        buf564 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg839_1, reinterpret_tensor(buf562, (16384, 128), (128, 1), 0), reinterpret_tensor(arg838_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf564)
        del arg838_1
        del arg839_1
        buf565 = reinterpret_tensor(buf562, (16384, 128), (128, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg841_1, reinterpret_tensor(buf560, (16384, 512), (512, 1), 0), reinterpret_tensor(arg840_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf565)
        del arg840_1
        del arg841_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf566 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf563, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf564, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf565, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf563
        buf567 = buf566[0]
        del buf566
        buf571 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (16384, 128), (128, 1), 0), reinterpret_tensor(arg842_1, (128, 128), (1, 128), 0), out=buf571)
        del arg842_1
        buf572 = reinterpret_tensor(buf567, (16384, 128), (128, 1), 0); del buf567  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf560, (16384, 512), (512, 1), 0), reinterpret_tensor(arg856_1, (512, 128), (1, 512), 0), out=buf572)
        del arg856_1
        buf573 = reinterpret_tensor(buf571, (128, 128, 128), (16384, 128, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [mul_146, layer_input_73, add_276, mul_148, layer_outputs_199], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf573, arg843_1, buf572, arg857_1, arg859_1, arg858_1, arg845_1, arg844_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg843_1
        del arg844_1
        del arg845_1
        del arg857_1
        del arg858_1
        del arg859_1
        buf574 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf573, (16384, 128), (128, 1), 0), reinterpret_tensor(arg864_1, (128, 512), (1, 128), 0), out=buf574)
        del arg864_1
        buf575 = reinterpret_tensor(buf574, (128, 128, 512), (65536, 512, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_145], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf575, arg865_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg865_1
        buf576 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf575, (16384, 512), (512, 1), 0), reinterpret_tensor(arg866_1, (512, 128), (1, 512), 0), out=buf576)
        del arg866_1
        buf577 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [add_278, mul_149, layer_outputs_201], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf577, buf576, arg867_1, arg869_1, arg868_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg867_1
        del arg868_1
        del arg869_1
        buf578 = reinterpret_tensor(buf575, (16384, 512), (512, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf577, (16384, 128), (128, 1), 0), reinterpret_tensor(arg870_1, (128, 512), (1, 128), 0), out=buf578)
        del arg870_1
        buf579 = reinterpret_tensor(buf578, (128, 128, 512), (65536, 512, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_147], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf579, arg871_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg871_1
        buf580 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf579, (16384, 512), (512, 1), 0), reinterpret_tensor(arg872_1, (512, 128), (1, 512), 0), out=buf580)
        del arg872_1
        buf581 = buf577; del buf577  # reuse
        # Topologically Sorted Source Nodes: [add_280, mul_150, layer_outputs_203], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf581, buf580, arg873_1, arg875_1, arg874_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg873_1
        del arg874_1
        del arg875_1
        buf582 = reinterpret_tensor(buf579, (16384, 512), (512, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf581, (16384, 128), (128, 1), 0), reinterpret_tensor(arg876_1, (128, 512), (1, 128), 0), out=buf582)
        del arg876_1
        buf583 = reinterpret_tensor(buf582, (128, 128, 512), (65536, 512, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_149], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf583, arg877_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg877_1
        buf584 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf583, (16384, 512), (512, 1), 0), reinterpret_tensor(arg878_1, (512, 128), (1, 512), 0), out=buf584)
        del arg878_1
        buf585 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [add_282, mul_151, layer_outputs_205], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf585, buf584, arg879_1, arg881_1, arg880_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg879_1
        del arg880_1
        del arg881_1
        buf586 = reinterpret_tensor(buf583, (16384, 512), (512, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf585, (16384, 128), (128, 1), 0), reinterpret_tensor(arg846_1, (128, 512), (1, 128), 0), out=buf586)
        del arg846_1
        buf587 = reinterpret_tensor(buf586, (128, 128, 512), (65536, 512, 1), 0); del buf586  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_151], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf587, arg847_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg847_1
        buf588 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf587, (16384, 512), (512, 1), 0), reinterpret_tensor(arg848_1, (512, 128), (1, 512), 0), out=buf588)
        del arg848_1
        buf589 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [add_284, mul_152, layer_output_37], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf589, buf588, arg849_1, arg851_1, arg850_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg849_1
        del arg850_1
        del arg851_1
        buf590 = reinterpret_tensor(buf587, (16384, 512), (512, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf589, (16384, 128), (128, 1), 0), reinterpret_tensor(arg852_1, (128, 512), (1, 128), 0), out=buf590)
        del arg852_1
        buf591 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [add_286, mul_153, layer_outputs_208], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf591, buf590, arg853_1, arg855_1, arg854_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg853_1
        del arg854_1
        del arg855_1
        buf592 = reinterpret_tensor(buf589, (16384, 128), (128, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf591, (16384, 512), (512, 1), 0), reinterpret_tensor(arg906_1, (512, 128), (1, 512), 0), out=buf592)
        del arg906_1
        buf593 = reinterpret_tensor(buf592, (128, 128, 128), (16384, 128, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [mul_155, layer_input_79], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf593, arg907_1, arg909_1, arg908_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg907_1
        del arg908_1
        del arg909_1
        buf594 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg883_1, reinterpret_tensor(buf593, (16384, 128), (128, 1), 0), reinterpret_tensor(arg882_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf594)
        del arg882_1
        del arg883_1
        buf595 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg885_1, reinterpret_tensor(buf593, (16384, 128), (128, 1), 0), reinterpret_tensor(arg884_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf595)
        del arg884_1
        del arg885_1
        buf596 = reinterpret_tensor(buf593, (16384, 128), (128, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg887_1, reinterpret_tensor(buf591, (16384, 512), (512, 1), 0), reinterpret_tensor(arg886_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf596)
        del arg886_1
        del arg887_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf597 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf594, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf595, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf596, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf594
        buf598 = buf597[0]
        del buf597
        buf602 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (16384, 128), (128, 1), 0), reinterpret_tensor(arg888_1, (128, 128), (1, 128), 0), out=buf602)
        del arg888_1
        buf603 = reinterpret_tensor(buf598, (16384, 128), (128, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf591, (16384, 512), (512, 1), 0), reinterpret_tensor(arg902_1, (512, 128), (1, 512), 0), out=buf603)
        del arg902_1
        buf604 = reinterpret_tensor(buf602, (128, 128, 128), (16384, 128, 1), 0); del buf602  # reuse
        # Topologically Sorted Source Nodes: [mul_154, layer_input_77, add_291, mul_156, layer_outputs_210], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf604, arg889_1, buf603, arg903_1, arg905_1, arg904_1, arg891_1, arg890_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg889_1
        del arg890_1
        del arg891_1
        del arg903_1
        del arg904_1
        del arg905_1
        buf605 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf604, (16384, 128), (128, 1), 0), reinterpret_tensor(arg910_1, (128, 512), (1, 128), 0), out=buf605)
        del arg910_1
        buf606 = reinterpret_tensor(buf605, (128, 128, 512), (65536, 512, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_153], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf606, arg911_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg911_1
        buf607 = buf603; del buf603  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf606, (16384, 512), (512, 1), 0), reinterpret_tensor(arg912_1, (512, 128), (1, 512), 0), out=buf607)
        del arg912_1
        buf608 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [add_293, mul_157, layer_outputs_212], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf608, buf607, arg913_1, arg915_1, arg914_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg913_1
        del arg914_1
        del arg915_1
        buf609 = reinterpret_tensor(buf606, (16384, 512), (512, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf608, (16384, 128), (128, 1), 0), reinterpret_tensor(arg916_1, (128, 512), (1, 128), 0), out=buf609)
        del arg916_1
        buf610 = reinterpret_tensor(buf609, (128, 128, 512), (65536, 512, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_155], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf610, arg917_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg917_1
        buf611 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf610, (16384, 512), (512, 1), 0), reinterpret_tensor(arg918_1, (512, 128), (1, 512), 0), out=buf611)
        del arg918_1
        buf612 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [add_295, mul_158, layer_outputs_214], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf612, buf611, arg919_1, arg921_1, arg920_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg919_1
        del arg920_1
        del arg921_1
        buf613 = reinterpret_tensor(buf610, (16384, 512), (512, 1), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf612, (16384, 128), (128, 1), 0), reinterpret_tensor(arg922_1, (128, 512), (1, 128), 0), out=buf613)
        del arg922_1
        buf614 = reinterpret_tensor(buf613, (128, 128, 512), (65536, 512, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_157], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf614, arg923_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg923_1
        buf615 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf614, (16384, 512), (512, 1), 0), reinterpret_tensor(arg924_1, (512, 128), (1, 512), 0), out=buf615)
        del arg924_1
        buf616 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [add_297, mul_159, layer_outputs_216], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf616, buf615, arg925_1, arg927_1, arg926_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg925_1
        del arg926_1
        del arg927_1
        buf617 = reinterpret_tensor(buf614, (16384, 512), (512, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf616, (16384, 128), (128, 1), 0), reinterpret_tensor(arg892_1, (128, 512), (1, 128), 0), out=buf617)
        del arg892_1
        buf618 = reinterpret_tensor(buf617, (128, 128, 512), (65536, 512, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_159], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf618, arg893_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg893_1
        buf619 = buf615; del buf615  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf618, (16384, 512), (512, 1), 0), reinterpret_tensor(arg894_1, (512, 128), (1, 512), 0), out=buf619)
        del arg894_1
        buf620 = buf616; del buf616  # reuse
        # Topologically Sorted Source Nodes: [add_299, mul_160, layer_output_39], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf620, buf619, arg895_1, arg897_1, arg896_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg895_1
        del arg896_1
        del arg897_1
        buf621 = reinterpret_tensor(buf618, (16384, 512), (512, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf620, (16384, 128), (128, 1), 0), reinterpret_tensor(arg898_1, (128, 512), (1, 128), 0), out=buf621)
        del arg898_1
        buf622 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [add_301, mul_161, layer_outputs_219], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf622, buf621, arg899_1, arg901_1, arg900_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg899_1
        del arg900_1
        del arg901_1
        buf623 = reinterpret_tensor(buf620, (16384, 128), (128, 1), 0); del buf620  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf622, (16384, 512), (512, 1), 0), reinterpret_tensor(arg952_1, (512, 128), (1, 512), 0), out=buf623)
        del arg952_1
        buf624 = reinterpret_tensor(buf623, (128, 128, 128), (16384, 128, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [mul_163, layer_input_83], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf624, arg953_1, arg955_1, arg954_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg953_1
        del arg954_1
        del arg955_1
        buf625 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg929_1, reinterpret_tensor(buf624, (16384, 128), (128, 1), 0), reinterpret_tensor(arg928_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf625)
        del arg928_1
        del arg929_1
        buf626 = buf595; del buf595  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg931_1, reinterpret_tensor(buf624, (16384, 128), (128, 1), 0), reinterpret_tensor(arg930_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf626)
        del arg930_1
        del arg931_1
        buf627 = reinterpret_tensor(buf624, (16384, 128), (128, 1), 0); del buf624  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg933_1, reinterpret_tensor(buf622, (16384, 512), (512, 1), 0), reinterpret_tensor(arg932_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf627)
        del arg932_1
        del arg933_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf628 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf625, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf626, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf627, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf625
        buf629 = buf628[0]
        del buf628
        buf633 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf629, (16384, 128), (128, 1), 0), reinterpret_tensor(arg934_1, (128, 128), (1, 128), 0), out=buf633)
        del arg934_1
        buf634 = reinterpret_tensor(buf629, (16384, 128), (128, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf622, (16384, 512), (512, 1), 0), reinterpret_tensor(arg948_1, (512, 128), (1, 512), 0), out=buf634)
        del arg948_1
        buf635 = reinterpret_tensor(buf633, (128, 128, 128), (16384, 128, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [mul_162, layer_input_81, add_306, mul_164, layer_outputs_221], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf635, arg935_1, buf634, arg949_1, arg951_1, arg950_1, arg937_1, arg936_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg935_1
        del arg936_1
        del arg937_1
        del arg949_1
        del arg950_1
        del arg951_1
        buf636 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf635, (16384, 128), (128, 1), 0), reinterpret_tensor(arg956_1, (128, 512), (1, 128), 0), out=buf636)
        del arg956_1
        buf637 = reinterpret_tensor(buf636, (128, 128, 512), (65536, 512, 1), 0); del buf636  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_161], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf637, arg957_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg957_1
        buf638 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf637, (16384, 512), (512, 1), 0), reinterpret_tensor(arg958_1, (512, 128), (1, 512), 0), out=buf638)
        del arg958_1
        buf639 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [add_308, mul_165, layer_outputs_223], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf639, buf638, arg959_1, arg961_1, arg960_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg959_1
        del arg960_1
        del arg961_1
        buf640 = reinterpret_tensor(buf637, (16384, 512), (512, 1), 0); del buf637  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf639, (16384, 128), (128, 1), 0), reinterpret_tensor(arg962_1, (128, 512), (1, 128), 0), out=buf640)
        del arg962_1
        buf641 = reinterpret_tensor(buf640, (128, 128, 512), (65536, 512, 1), 0); del buf640  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_163], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf641, arg963_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg963_1
        buf642 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf641, (16384, 512), (512, 1), 0), reinterpret_tensor(arg964_1, (512, 128), (1, 512), 0), out=buf642)
        del arg964_1
        buf643 = buf639; del buf639  # reuse
        # Topologically Sorted Source Nodes: [add_310, mul_166, layer_outputs_225], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf643, buf642, arg965_1, arg967_1, arg966_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg965_1
        del arg966_1
        del arg967_1
        buf644 = reinterpret_tensor(buf641, (16384, 512), (512, 1), 0); del buf641  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf643, (16384, 128), (128, 1), 0), reinterpret_tensor(arg968_1, (128, 512), (1, 128), 0), out=buf644)
        del arg968_1
        buf645 = reinterpret_tensor(buf644, (128, 128, 512), (65536, 512, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_165], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf645, arg969_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg969_1
        buf646 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf645, (16384, 512), (512, 1), 0), reinterpret_tensor(arg970_1, (512, 128), (1, 512), 0), out=buf646)
        del arg970_1
        buf647 = buf643; del buf643  # reuse
        # Topologically Sorted Source Nodes: [add_312, mul_167, layer_outputs_227], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf647, buf646, arg971_1, arg973_1, arg972_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg971_1
        del arg972_1
        del arg973_1
        buf648 = reinterpret_tensor(buf645, (16384, 512), (512, 1), 0); del buf645  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf647, (16384, 128), (128, 1), 0), reinterpret_tensor(arg938_1, (128, 512), (1, 128), 0), out=buf648)
        del arg938_1
        buf649 = reinterpret_tensor(buf648, (128, 128, 512), (65536, 512, 1), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_167], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf649, arg939_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg939_1
        buf650 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf649, (16384, 512), (512, 1), 0), reinterpret_tensor(arg940_1, (512, 128), (1, 512), 0), out=buf650)
        del arg940_1
        buf651 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [add_314, mul_168, layer_output_41], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf651, buf650, arg941_1, arg943_1, arg942_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg941_1
        del arg942_1
        del arg943_1
        buf652 = reinterpret_tensor(buf649, (16384, 512), (512, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf651, (16384, 128), (128, 1), 0), reinterpret_tensor(arg944_1, (128, 512), (1, 128), 0), out=buf652)
        del arg944_1
        buf653 = buf622; del buf622  # reuse
        # Topologically Sorted Source Nodes: [add_316, mul_169, layer_outputs_230], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf653, buf652, arg945_1, arg947_1, arg946_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg945_1
        del arg946_1
        del arg947_1
        buf654 = reinterpret_tensor(buf651, (16384, 128), (128, 1), 0); del buf651  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (16384, 512), (512, 1), 0), reinterpret_tensor(arg998_1, (512, 128), (1, 512), 0), out=buf654)
        del arg998_1
        buf655 = reinterpret_tensor(buf654, (128, 128, 128), (16384, 128, 1), 0); del buf654  # reuse
        # Topologically Sorted Source Nodes: [mul_171, layer_input_87], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf655, arg999_1, arg1001_1, arg1000_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1000_1
        del arg1001_1
        del arg999_1
        buf656 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg975_1, reinterpret_tensor(buf655, (16384, 128), (128, 1), 0), reinterpret_tensor(arg974_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf656)
        del arg974_1
        del arg975_1
        buf657 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg977_1, reinterpret_tensor(buf655, (16384, 128), (128, 1), 0), reinterpret_tensor(arg976_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf657)
        del arg976_1
        del arg977_1
        buf658 = reinterpret_tensor(buf655, (16384, 128), (128, 1), 0); del buf655  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg979_1, reinterpret_tensor(buf653, (16384, 512), (512, 1), 0), reinterpret_tensor(arg978_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf658)
        del arg978_1
        del arg979_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf659 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf656, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf657, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf658, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf656
        buf660 = buf659[0]
        del buf659
        buf664 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf660, (16384, 128), (128, 1), 0), reinterpret_tensor(arg980_1, (128, 128), (1, 128), 0), out=buf664)
        del arg980_1
        buf665 = reinterpret_tensor(buf660, (16384, 128), (128, 1), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf653, (16384, 512), (512, 1), 0), reinterpret_tensor(arg994_1, (512, 128), (1, 512), 0), out=buf665)
        del arg994_1
        buf666 = reinterpret_tensor(buf664, (128, 128, 128), (16384, 128, 1), 0); del buf664  # reuse
        # Topologically Sorted Source Nodes: [mul_170, layer_input_85, add_321, mul_172, layer_outputs_232], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf666, arg981_1, buf665, arg995_1, arg997_1, arg996_1, arg983_1, arg982_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg981_1
        del arg982_1
        del arg983_1
        del arg995_1
        del arg996_1
        del arg997_1
        buf667 = buf652; del buf652  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf666, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1002_1, (128, 512), (1, 128), 0), out=buf667)
        del arg1002_1
        buf668 = reinterpret_tensor(buf667, (128, 128, 512), (65536, 512, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_169], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf668, arg1003_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1003_1
        buf669 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf668, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1004_1, (512, 128), (1, 512), 0), out=buf669)
        del arg1004_1
        buf670 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [add_323, mul_173, layer_outputs_234], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf670, buf669, arg1005_1, arg1007_1, arg1006_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1005_1
        del arg1006_1
        del arg1007_1
        buf671 = reinterpret_tensor(buf668, (16384, 512), (512, 1), 0); del buf668  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf670, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1008_1, (128, 512), (1, 128), 0), out=buf671)
        del arg1008_1
        buf672 = reinterpret_tensor(buf671, (128, 128, 512), (65536, 512, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_171], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf672, arg1009_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1009_1
        buf673 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf672, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1010_1, (512, 128), (1, 512), 0), out=buf673)
        del arg1010_1
        buf674 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [add_325, mul_174, layer_outputs_236], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf674, buf673, arg1011_1, arg1013_1, arg1012_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1011_1
        del arg1012_1
        del arg1013_1
        buf675 = reinterpret_tensor(buf672, (16384, 512), (512, 1), 0); del buf672  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf674, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1014_1, (128, 512), (1, 128), 0), out=buf675)
        del arg1014_1
        buf676 = reinterpret_tensor(buf675, (128, 128, 512), (65536, 512, 1), 0); del buf675  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_173], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf676, arg1015_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1015_1
        buf677 = buf673; del buf673  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf676, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1016_1, (512, 128), (1, 512), 0), out=buf677)
        del arg1016_1
        buf678 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [add_327, mul_175, layer_outputs_238], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf678, buf677, arg1017_1, arg1019_1, arg1018_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1017_1
        del arg1018_1
        del arg1019_1
        buf679 = reinterpret_tensor(buf676, (16384, 512), (512, 1), 0); del buf676  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf678, (16384, 128), (128, 1), 0), reinterpret_tensor(arg984_1, (128, 512), (1, 128), 0), out=buf679)
        del arg984_1
        buf680 = reinterpret_tensor(buf679, (128, 128, 512), (65536, 512, 1), 0); del buf679  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_175], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf680, arg985_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg985_1
        buf681 = buf677; del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf680, (16384, 512), (512, 1), 0), reinterpret_tensor(arg986_1, (512, 128), (1, 512), 0), out=buf681)
        del arg986_1
        buf682 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [add_329, mul_176, layer_output_43], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf682, buf681, arg987_1, arg989_1, arg988_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg987_1
        del arg988_1
        del arg989_1
        buf683 = reinterpret_tensor(buf680, (16384, 512), (512, 1), 0); del buf680  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf682, (16384, 128), (128, 1), 0), reinterpret_tensor(arg990_1, (128, 512), (1, 128), 0), out=buf683)
        del arg990_1
        buf684 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [add_331, mul_177, layer_outputs_241], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf684, buf683, arg991_1, arg993_1, arg992_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg991_1
        del arg992_1
        del arg993_1
        buf685 = reinterpret_tensor(buf682, (16384, 128), (128, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1044_1, (512, 128), (1, 512), 0), out=buf685)
        del arg1044_1
        buf686 = reinterpret_tensor(buf685, (128, 128, 128), (16384, 128, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [mul_179, layer_input_91], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf686, arg1045_1, arg1047_1, arg1046_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1045_1
        del arg1046_1
        del arg1047_1
        buf687 = buf681; del buf681  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1021_1, reinterpret_tensor(buf686, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1020_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf687)
        del arg1020_1
        del arg1021_1
        buf688 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1023_1, reinterpret_tensor(buf686, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1022_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf688)
        del arg1022_1
        del arg1023_1
        buf689 = reinterpret_tensor(buf686, (16384, 128), (128, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1025_1, reinterpret_tensor(buf684, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1024_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf689)
        del arg1024_1
        del arg1025_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf690 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf687, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf688, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf689, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf687
        buf691 = buf690[0]
        del buf690
        buf695 = buf689; del buf689  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf691, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1026_1, (128, 128), (1, 128), 0), out=buf695)
        del arg1026_1
        buf696 = reinterpret_tensor(buf691, (16384, 128), (128, 1), 0); del buf691  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf684, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1040_1, (512, 128), (1, 512), 0), out=buf696)
        del arg1040_1
        buf697 = reinterpret_tensor(buf695, (128, 128, 128), (16384, 128, 1), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [mul_178, layer_input_89, add_336, mul_180, layer_outputs_243], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf697, arg1027_1, buf696, arg1041_1, arg1043_1, arg1042_1, arg1029_1, arg1028_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1027_1
        del arg1028_1
        del arg1029_1
        del arg1041_1
        del arg1042_1
        del arg1043_1
        buf698 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf697, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1048_1, (128, 512), (1, 128), 0), out=buf698)
        del arg1048_1
        buf699 = reinterpret_tensor(buf698, (128, 128, 512), (65536, 512, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_177], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf699, arg1049_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1049_1
        buf700 = buf696; del buf696  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf699, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1050_1, (512, 128), (1, 512), 0), out=buf700)
        del arg1050_1
        buf701 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [add_338, mul_181, layer_outputs_245], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf701, buf700, arg1051_1, arg1053_1, arg1052_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1051_1
        del arg1052_1
        del arg1053_1
        buf702 = reinterpret_tensor(buf699, (16384, 512), (512, 1), 0); del buf699  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf701, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1054_1, (128, 512), (1, 128), 0), out=buf702)
        del arg1054_1
        buf703 = reinterpret_tensor(buf702, (128, 128, 512), (65536, 512, 1), 0); del buf702  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_179], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf703, arg1055_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1055_1
        buf704 = buf700; del buf700  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf703, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1056_1, (512, 128), (1, 512), 0), out=buf704)
        del arg1056_1
        buf705 = buf701; del buf701  # reuse
        # Topologically Sorted Source Nodes: [add_340, mul_182, layer_outputs_247], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf705, buf704, arg1057_1, arg1059_1, arg1058_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1057_1
        del arg1058_1
        del arg1059_1
        buf706 = reinterpret_tensor(buf703, (16384, 512), (512, 1), 0); del buf703  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1060_1, (128, 512), (1, 128), 0), out=buf706)
        del arg1060_1
        buf707 = reinterpret_tensor(buf706, (128, 128, 512), (65536, 512, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_181], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf707, arg1061_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1061_1
        buf708 = buf704; del buf704  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf707, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1062_1, (512, 128), (1, 512), 0), out=buf708)
        del arg1062_1
        buf709 = buf705; del buf705  # reuse
        # Topologically Sorted Source Nodes: [add_342, mul_183, layer_outputs_249], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf709, buf708, arg1063_1, arg1065_1, arg1064_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1063_1
        del arg1064_1
        del arg1065_1
        buf710 = reinterpret_tensor(buf707, (16384, 512), (512, 1), 0); del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf709, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1030_1, (128, 512), (1, 128), 0), out=buf710)
        del arg1030_1
        buf711 = reinterpret_tensor(buf710, (128, 128, 512), (65536, 512, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_183], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf711, arg1031_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1031_1
        buf712 = buf708; del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf711, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1032_1, (512, 128), (1, 512), 0), out=buf712)
        del arg1032_1
        buf713 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [add_344, mul_184, layer_output_45], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf713, buf712, arg1033_1, arg1035_1, arg1034_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1033_1
        del arg1034_1
        del arg1035_1
        buf714 = reinterpret_tensor(buf711, (16384, 512), (512, 1), 0); del buf711  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf713, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1036_1, (128, 512), (1, 128), 0), out=buf714)
        del arg1036_1
        buf715 = buf684; del buf684  # reuse
        # Topologically Sorted Source Nodes: [add_346, mul_185, layer_outputs_252], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf715, buf714, arg1037_1, arg1039_1, arg1038_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1037_1
        del arg1038_1
        del arg1039_1
        buf716 = reinterpret_tensor(buf713, (16384, 128), (128, 1), 0); del buf713  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf715, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1090_1, (512, 128), (1, 512), 0), out=buf716)
        del arg1090_1
        buf717 = reinterpret_tensor(buf716, (128, 128, 128), (16384, 128, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [mul_187, layer_input_95], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_2.run(buf717, arg1091_1, arg1093_1, arg1092_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1091_1
        del arg1092_1
        del arg1093_1
        buf718 = buf712; del buf712  # reuse
        # Topologically Sorted Source Nodes: [mixed_query_layer_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1067_1, reinterpret_tensor(buf717, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1066_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf718)
        del arg1066_1
        del arg1067_1
        buf719 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [mixed_key_layer_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1069_1, reinterpret_tensor(buf717, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1068_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf719)
        del arg1068_1
        del arg1069_1
        buf720 = reinterpret_tensor(buf717, (16384, 128), (128, 1), 0); del buf717  # reuse
        # Topologically Sorted Source Nodes: [mixed_value_layer_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1071_1, reinterpret_tensor(buf715, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1070_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf720)
        del arg1070_1
        del arg1071_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf721 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf718, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf719, (128, 4, 128, 32), (16384, 32, 128, 1), 0), reinterpret_tensor(buf720, (128, 4, 128, 32), (16384, 32, 128, 1), 0), None, False, scale=0.17677669529663687)
        del buf718
        del buf719
        buf722 = buf721[0]
        del buf721
        buf726 = buf720; del buf720  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf722, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1072_1, (128, 128), (1, 128), 0), out=buf726)
        del arg1072_1
        buf727 = reinterpret_tensor(buf722, (16384, 128), (128, 1), 0); del buf722  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf715, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1086_1, (512, 128), (1, 512), 0), out=buf727)
        del arg1086_1
        buf728 = reinterpret_tensor(buf726, (128, 128, 128), (16384, 128, 1), 0); del buf726  # reuse
        # Topologically Sorted Source Nodes: [mul_186, layer_input_93, add_351, mul_188, layer_outputs_254], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_3.run(buf728, arg1073_1, buf727, arg1087_1, arg1089_1, arg1088_1, arg1075_1, arg1074_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1073_1
        del arg1074_1
        del arg1075_1
        del arg1087_1
        del arg1088_1
        del arg1089_1
        buf729 = buf714; del buf714  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf728, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1094_1, (128, 512), (1, 128), 0), out=buf729)
        del arg1094_1
        buf730 = reinterpret_tensor(buf729, (128, 128, 512), (65536, 512, 1), 0); del buf729  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_185], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf730, arg1095_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1095_1
        buf731 = buf727; del buf727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf730, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1096_1, (512, 128), (1, 512), 0), out=buf731)
        del arg1096_1
        buf732 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [add_353, mul_189, layer_outputs_256], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf732, buf731, arg1097_1, arg1099_1, arg1098_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1097_1
        del arg1098_1
        del arg1099_1
        buf733 = reinterpret_tensor(buf730, (16384, 512), (512, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf732, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1100_1, (128, 512), (1, 128), 0), out=buf733)
        del arg1100_1
        buf734 = reinterpret_tensor(buf733, (128, 128, 512), (65536, 512, 1), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_187], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf734, arg1101_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1101_1
        buf735 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf734, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1102_1, (512, 128), (1, 512), 0), out=buf735)
        del arg1102_1
        buf736 = buf732; del buf732  # reuse
        # Topologically Sorted Source Nodes: [add_355, mul_190, layer_outputs_258], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf736, buf735, arg1103_1, arg1105_1, arg1104_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1103_1
        del arg1104_1
        del arg1105_1
        buf737 = reinterpret_tensor(buf734, (16384, 512), (512, 1), 0); del buf734  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf736, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1106_1, (128, 512), (1, 128), 0), out=buf737)
        del arg1106_1
        buf738 = reinterpret_tensor(buf737, (128, 128, 512), (65536, 512, 1), 0); del buf737  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_189], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf738, arg1107_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1107_1
        buf739 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf738, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1108_1, (512, 128), (1, 512), 0), out=buf739)
        del arg1108_1
        buf740 = buf736; del buf736  # reuse
        # Topologically Sorted Source Nodes: [add_357, mul_191, layer_outputs_260], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf740, buf739, arg1109_1, arg1111_1, arg1110_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1109_1
        del arg1110_1
        del arg1111_1
        buf741 = reinterpret_tensor(buf738, (16384, 512), (512, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf740, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1076_1, (128, 512), (1, 128), 0), out=buf741)
        del arg1076_1
        buf742 = reinterpret_tensor(buf741, (128, 128, 512), (65536, 512, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_191], Original ATen: [aten.relu]
        triton_poi_fused_relu_4.run(buf742, arg1077_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1077_1
        buf743 = buf739; del buf739  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf742, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1078_1, (512, 128), (1, 512), 0), out=buf743)
        del arg1078_1
        buf744 = buf740; del buf740  # reuse
        # Topologically Sorted Source Nodes: [add_359, mul_192, layer_output_47], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(buf744, buf743, arg1079_1, arg1081_1, arg1080_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg1079_1
        del arg1080_1
        del arg1081_1
        del buf743
        buf745 = reinterpret_tensor(buf742, (16384, 512), (512, 1), 0); del buf742  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf744, (16384, 128), (128, 1), 0), reinterpret_tensor(arg1082_1, (128, 512), (1, 128), 0), out=buf745)
        del arg1082_1
        del buf744
        buf746 = buf715; del buf715  # reuse
        # Topologically Sorted Source Nodes: [add_361, mul_193, layer_outputs_263], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_6.run(buf746, buf745, arg1083_1, arg1085_1, arg1084_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg1083_1
        del arg1084_1
        del arg1085_1
        buf747 = buf745; del buf745  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf746, (16384, 512), (512, 1), 0), reinterpret_tensor(arg1113_1, (512, 512), (1, 512), 0), out=buf747)
        del arg1113_1
        buf751 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_193, hidden_states_194], Original ATen: [aten.relu, aten.native_layer_norm]
        triton_per_fused_native_layer_norm_relu_8.run(buf747, arg1114_1, arg1115_1, arg1116_1, buf751, 16384, 512, grid=grid(16384), stream=stream0)
        del arg1114_1
        del arg1115_1
        del arg1116_1
        del buf747
        buf752 = empty_strided_cuda((512, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_poi_fused_9.run(arg1_1, arg1117_1, buf752, 15628288, grid=grid(15628288), stream=stream0)
        del arg1117_1
        del arg1_1
        buf753 = empty_strided_cuda((16384, 30524), (30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf751, (16384, 512), (512, 1), 0), buf752, out=buf753)
        del buf751
        del buf752
        buf754 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        buf755 = empty_strided_cuda((16384, 1), (1, 16384), torch.float32)
        buf760 = empty_strided_cuda((128, 128, 30522), (3907584, 30528, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_196, masked_lm_loss], Original ATen: [aten.add, aten._log_softmax]
        triton_red_fused__log_softmax_add_10.run(buf753, arg1118_1, buf754, buf755, buf760, 16384, 30522, grid=grid(16384), stream=stream0)
        buf756 = empty_strided_cuda((2, ), (1, ), torch.float32)
        buf758 = empty_strided_cuda((2, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_11.run(arg1119_1, buf753, arg1118_1, buf754, buf755, buf756, buf758, 2, 8192, grid=grid(2), stream=stream0)
        del arg1118_1
        del arg1119_1
        del buf753
        del buf754
        del buf755
        buf757 = empty_strided_cuda((), (), torch.float32)
        buf761 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [masked_lm_loss], Original ATen: [aten.nll_loss_forward]
        triton_per_fused_nll_loss_forward_12.run(buf761, buf756, buf758, 1, 2, grid=grid(1), stream=stream0)
        del buf756
        del buf758
    return (buf761, buf760, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1113_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((384, 30522), (30522, 1), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((30522, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1119_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForMaskedLM', benchmark_compiled_module)
