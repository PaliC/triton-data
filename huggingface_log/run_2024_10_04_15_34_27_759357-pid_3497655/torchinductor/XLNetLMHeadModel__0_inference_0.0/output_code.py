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


# kernel path: /tmp/torchinductor_sahanp/wk/cwksp66463bvzhlf5ameefti5nc6lnfhe3sfgcawirm32d22fdwz.py
# Topologically Sorted Source Nodes: [input_ids, word_emb_k], Original ATen: [aten.clone, aten.embedding]
# Source node to ATen node mapping:
#   input_ids => clone
#   word_emb_k => embedding
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %embedding : [num_users=5] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %clone), kwargs = {})
triton_poi_fused_clone_embedding_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_embedding_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 1024) % 8
    x2 = (xindex // 8192)
    x0 = xindex % 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (512*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32000, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32000), "index out of bounds: 0 <= tmp4 < 32000")
    tmp6 = tl.load(in_ptr1 + (x0 + (1024*tmp4)), None)
    tl.store(out_ptr0 + (x4), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x3/cx342inz7k6ppheezyfutvcazozh3zi2i767ti46di7h3astdj4d.py
# Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %arg8_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %arg7_1), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


cpp_fused_cos_mul_sin_2 = async_compile.cpp_pybinding(['float*', 'float*'], '''
#include "/tmp/torchinductor_sahanp/z4/cz4j2mmotlx3z2b7u4fbjtdt4x6plhd67ljwzg5bk7ekv4xz6y7q.h"
extern "C"  void kernel(float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(1024L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = 2L*x1;
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp2 = at::vec::Vectorized<float>::arange(tmp1, 2);
                auto tmp3 = static_cast<float>(0.0009765625);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = static_cast<float>(10000.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp7.pow(tmp5);
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = 512L + ((-1L)*x0);
                auto tmp14 = c10::convert<float>(tmp13);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp15 * tmp12;
                auto tmp17 = tmp16.sin();
                tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + (1024L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(1024L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(16L))
            {
                auto tmp0 = 2L*x1;
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp2 = at::vec::Vectorized<float>::arange(tmp1, 2);
                auto tmp3 = static_cast<float>(0.0009765625);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = static_cast<float>(10000.0);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp7.pow(tmp5);
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = 512L + ((-1L)*x0);
                auto tmp14 = c10::convert<float>(tmp13);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp15 * tmp12;
                auto tmp17 = tmp16.cos();
                tmp17.store(out_ptr1 + static_cast<int64_t>(x1 + (1024L*x0)));
            }
        }
    }
}
''')


# kernel path: /tmp/torchinductor_sahanp/bp/cbpfrcl6ewadteb73x5rw733qqpzahh7vgbuj3wyxmoy2j546jjp.py
# Topologically Sorted Source Nodes: [pos_emb_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   pos_emb_3 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=24] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%device_put, torch.float32), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5a/c5a77fc4hec3wtnz5qhftzpwiu63x2bjrtp5kjaqhisqxaj7daet.py
# Topologically Sorted Source Nodes: [x_3, add_2, add_3, attn_prob], Original ATen: [aten.index_select, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   attn_prob => div_1, exp, sum_1
#   x_3 => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_25, [None, None, None, %iota_2]), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %index), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0), kwargs = {})
#   %mul_tensor_46 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 1), kwargs = {})
#   %amax_default_23 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_46, [3], True), kwargs = {})
#   %sub_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_46, %amax_default_23), kwargs = {})
#   %mul_tensor_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_23, 0.125), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_47,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [3], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_red_fused__softmax_add_index_select_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 16
    x2 = (xindex // 8192)
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp10 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = 0.0
        tmp14 = tmp12 + tmp13
        tmp15 = 1.0
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16 - tmp8
        tmp18 = 0.125
        tmp19 = tmp17 * tmp18
        tmp20 = tl_math.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp24 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp24 + tmp25
        tmp27 = 0.0
        tmp28 = tmp26 + tmp27
        tmp29 = 1.0
        tmp30 = tmp28 * tmp29
        tmp31 = tmp30 - tmp8
        tmp32 = 0.125
        tmp33 = tmp31 * tmp32
        tmp34 = tl_math.exp(tmp33)
        tmp35 = tmp34 / tmp22
        tl.store(out_ptr2 + (r3 + (512*x4)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/my/cmy2x7ibpqwhhkfi3w6qtpwyjbh3fxv6hncrl6h5ap7msfzbzn54.py
# Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_out => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_40,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64) % 8
    y2 = (yindex // 512)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*y2) + (32768*x3) + (524288*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gn/cgnf6gu2sluunlpzwiktihgzqnvul4oex3ell4vsi6hoitosyojt.py
# Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_out => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x1 = xindex % 1024
    x2 = (xindex // 1024)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (1024*x1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3 + (16384*y0)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tj/ctjmaesbm7ic3iqa4othn2mahga66ksxjksbmtcrzc5gkoyzxkih.py
# Topologically Sorted Source Nodes: [attn_out_2, output], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   attn_out_2 => add_4
#   output => add_5, add_6, mul_3, mul_4, rsqrt, sub_1, var_mean
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_33, %embedding), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg9_1), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg10_1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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
    tmp23 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 - tmp10
    tmp17 = 1024.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-12
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qf/cqfaxsplq64lye2c64skfbmydlr2ku6nq6niwzmglz5dmbxaazt2.py
# Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   output_2 => add_7, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_7), kwargs = {})
triton_poi_fused_gelu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_sahanp/bk/cbkanwol74irmtkqkad5se5xheo2kje6q4tfe2fpgff3eivnmas2.py
# Topologically Sorted Source Nodes: [add_5, output_6], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_5 => add_8
#   output_6 => add_10, add_9, mul_8, mul_9, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_37, %add_6), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_3), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-12), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg11_1), kwargs = {})
#   %add_10 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg12_1), kwargs = {})
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), None)
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp21 = 1e-12
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q6/cq6bawefof22i3hwlmpv4in3axibc5lr447shkibx673mmsyj4vd.py
# Topologically Sorted Source Nodes: [add_143, output_167, output_169], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
# Source node to ATen node mapping:
#   add_143 => add_261
#   output_167 => var_mean_47
#   output_169 => clone_148
# Graph fragment:
#   %add_261 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_911, %add_259), kwargs = {})
#   %var_mean_47 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_261, [2]), kwargs = {correction: 0, keepdim: True})
#   %clone_148 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1011,), kwargs = {memory_format: torch.contiguous_format})
triton_per_fused_add_clone_native_layer_norm_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_10', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    x2 = xindex % 8
    x3 = (xindex // 8)
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), None)
    tmp25 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp21 = 1e-12
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (1024*x3) + (524288*x2)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l2/cl236iedoffof6qnfqmfkak5rn72tlaaripp6krjmrlecekjn4qc.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   loss => amax_24, exp_24, sub_72, sum_25
# Graph fragment:
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_914, [1], True), kwargs = {})
#   %sub_72 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_914, %amax_24), kwargs = {})
#   %exp_24 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_72,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [1], True), kwargs = {})
triton_red_fused__log_softmax_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 32000
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
        tmp0 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, eviction_policy='evict_last', other=0.0)
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
        tmp4 = tl.load(in_ptr0 + (r1 + (32000*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp4 - tmp2
        tmp6 = tl_math.exp(tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mi/cmiamgopoywwr2e7cwnwg74hracwqrmraykwsobgr2o3yftrwlp5.py
# Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   loss => convert_element_type_3, div_25, full_default_1, ne_1, ne_2, neg, sum_26, sum_27, where_1
# Graph fragment:
#   %ne_1 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_915, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg, %full_default_1), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
#   %ne_2 : [num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_915, -100), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_2,), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_26, torch.float32), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_27, %convert_element_type_3), kwargs = {})
triton_red_fused_nll_loss_forward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'AF2498D67158CAADADFFD49D59358CB8F5E4B1FCD1FBD49EE9B7C2D9E5D02859', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp5 = tl.full([XBLOCK, RBLOCK], 32000, tl.int32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp4 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp4)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 32000)) | ~(rmask), "index out of bounds: 0 <= tmp8 < 32000")
        tmp10 = tl.load(in_ptr1 + (tmp8 + (32000*r0)), rmask, eviction_policy='evict_last', other=0.0)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 512), (512, 1))
    assert_size_stride(arg1_1, (32000, 1024), (1024, 1))
    assert_size_stride(arg2_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg3_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg4_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg5_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg6_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg7_1, (16, 64), (64, 1))
    assert_size_stride(arg8_1, (16, 64), (64, 1))
    assert_size_stride(arg9_1, (1024, ), (1, ))
    assert_size_stride(arg10_1, (1024, ), (1, ))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (1024, ), (1, ))
    assert_size_stride(arg13_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg16_1, (1024, ), (1, ))
    assert_size_stride(arg17_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg18_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg19_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg20_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg21_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg22_1, (16, 64), (64, 1))
    assert_size_stride(arg23_1, (16, 64), (64, 1))
    assert_size_stride(arg24_1, (1024, ), (1, ))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (1024, ), (1, ))
    assert_size_stride(arg28_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg29_1, (4096, ), (1, ))
    assert_size_stride(arg30_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg31_1, (1024, ), (1, ))
    assert_size_stride(arg32_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg33_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg34_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg35_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg36_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg37_1, (16, 64), (64, 1))
    assert_size_stride(arg38_1, (16, 64), (64, 1))
    assert_size_stride(arg39_1, (1024, ), (1, ))
    assert_size_stride(arg40_1, (1024, ), (1, ))
    assert_size_stride(arg41_1, (1024, ), (1, ))
    assert_size_stride(arg42_1, (1024, ), (1, ))
    assert_size_stride(arg43_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg44_1, (4096, ), (1, ))
    assert_size_stride(arg45_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg48_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg49_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg50_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg51_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg52_1, (16, 64), (64, 1))
    assert_size_stride(arg53_1, (16, 64), (64, 1))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg59_1, (4096, ), (1, ))
    assert_size_stride(arg60_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg61_1, (1024, ), (1, ))
    assert_size_stride(arg62_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg63_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg64_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg65_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg66_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg67_1, (16, 64), (64, 1))
    assert_size_stride(arg68_1, (16, 64), (64, 1))
    assert_size_stride(arg69_1, (1024, ), (1, ))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (1024, ), (1, ))
    assert_size_stride(arg72_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg74_1, (4096, ), (1, ))
    assert_size_stride(arg75_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg78_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg79_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg80_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg81_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg82_1, (16, 64), (64, 1))
    assert_size_stride(arg83_1, (16, 64), (64, 1))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg89_1, (4096, ), (1, ))
    assert_size_stride(arg90_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg93_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg94_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg95_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg96_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg97_1, (16, 64), (64, 1))
    assert_size_stride(arg98_1, (16, 64), (64, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (1024, ), (1, ))
    assert_size_stride(arg103_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg104_1, (4096, ), (1, ))
    assert_size_stride(arg105_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg108_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg109_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg110_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg111_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg112_1, (16, 64), (64, 1))
    assert_size_stride(arg113_1, (16, 64), (64, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg119_1, (4096, ), (1, ))
    assert_size_stride(arg120_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg123_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg124_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg125_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg126_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg127_1, (16, 64), (64, 1))
    assert_size_stride(arg128_1, (16, 64), (64, 1))
    assert_size_stride(arg129_1, (1024, ), (1, ))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg134_1, (4096, ), (1, ))
    assert_size_stride(arg135_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg138_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg139_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg140_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg141_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg142_1, (16, 64), (64, 1))
    assert_size_stride(arg143_1, (16, 64), (64, 1))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg149_1, (4096, ), (1, ))
    assert_size_stride(arg150_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg153_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg154_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg155_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg156_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg157_1, (16, 64), (64, 1))
    assert_size_stride(arg158_1, (16, 64), (64, 1))
    assert_size_stride(arg159_1, (1024, ), (1, ))
    assert_size_stride(arg160_1, (1024, ), (1, ))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg164_1, (4096, ), (1, ))
    assert_size_stride(arg165_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg168_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg169_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg170_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg171_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg172_1, (16, 64), (64, 1))
    assert_size_stride(arg173_1, (16, 64), (64, 1))
    assert_size_stride(arg174_1, (1024, ), (1, ))
    assert_size_stride(arg175_1, (1024, ), (1, ))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (1024, ), (1, ))
    assert_size_stride(arg178_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg179_1, (4096, ), (1, ))
    assert_size_stride(arg180_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg183_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg184_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg185_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg186_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg187_1, (16, 64), (64, 1))
    assert_size_stride(arg188_1, (16, 64), (64, 1))
    assert_size_stride(arg189_1, (1024, ), (1, ))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (1024, ), (1, ))
    assert_size_stride(arg193_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg194_1, (4096, ), (1, ))
    assert_size_stride(arg195_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg198_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg199_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg200_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg201_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg202_1, (16, 64), (64, 1))
    assert_size_stride(arg203_1, (16, 64), (64, 1))
    assert_size_stride(arg204_1, (1024, ), (1, ))
    assert_size_stride(arg205_1, (1024, ), (1, ))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg209_1, (4096, ), (1, ))
    assert_size_stride(arg210_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg213_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg214_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg215_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg216_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg217_1, (16, 64), (64, 1))
    assert_size_stride(arg218_1, (16, 64), (64, 1))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (1024, ), (1, ))
    assert_size_stride(arg222_1, (1024, ), (1, ))
    assert_size_stride(arg223_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg224_1, (4096, ), (1, ))
    assert_size_stride(arg225_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg228_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg229_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg230_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg231_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg232_1, (16, 64), (64, 1))
    assert_size_stride(arg233_1, (16, 64), (64, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg239_1, (4096, ), (1, ))
    assert_size_stride(arg240_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg241_1, (1024, ), (1, ))
    assert_size_stride(arg242_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg243_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg244_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg245_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg246_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg247_1, (16, 64), (64, 1))
    assert_size_stride(arg248_1, (16, 64), (64, 1))
    assert_size_stride(arg249_1, (1024, ), (1, ))
    assert_size_stride(arg250_1, (1024, ), (1, ))
    assert_size_stride(arg251_1, (1024, ), (1, ))
    assert_size_stride(arg252_1, (1024, ), (1, ))
    assert_size_stride(arg253_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg254_1, (4096, ), (1, ))
    assert_size_stride(arg255_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg256_1, (1024, ), (1, ))
    assert_size_stride(arg257_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg258_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg259_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg260_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg261_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg262_1, (16, 64), (64, 1))
    assert_size_stride(arg263_1, (16, 64), (64, 1))
    assert_size_stride(arg264_1, (1024, ), (1, ))
    assert_size_stride(arg265_1, (1024, ), (1, ))
    assert_size_stride(arg266_1, (1024, ), (1, ))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg269_1, (4096, ), (1, ))
    assert_size_stride(arg270_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg273_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg274_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg275_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg276_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg277_1, (16, 64), (64, 1))
    assert_size_stride(arg278_1, (16, 64), (64, 1))
    assert_size_stride(arg279_1, (1024, ), (1, ))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg284_1, (4096, ), (1, ))
    assert_size_stride(arg285_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg286_1, (1024, ), (1, ))
    assert_size_stride(arg287_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg288_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg289_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg290_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg291_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg292_1, (16, 64), (64, 1))
    assert_size_stride(arg293_1, (16, 64), (64, 1))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (1024, ), (1, ))
    assert_size_stride(arg297_1, (1024, ), (1, ))
    assert_size_stride(arg298_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg299_1, (4096, ), (1, ))
    assert_size_stride(arg300_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg303_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg304_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg305_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg306_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg307_1, (16, 64), (64, 1))
    assert_size_stride(arg308_1, (16, 64), (64, 1))
    assert_size_stride(arg309_1, (1024, ), (1, ))
    assert_size_stride(arg310_1, (1024, ), (1, ))
    assert_size_stride(arg311_1, (1024, ), (1, ))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg314_1, (4096, ), (1, ))
    assert_size_stride(arg315_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg316_1, (1024, ), (1, ))
    assert_size_stride(arg317_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg318_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg319_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg320_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg321_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg322_1, (16, 64), (64, 1))
    assert_size_stride(arg323_1, (16, 64), (64, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg329_1, (4096, ), (1, ))
    assert_size_stride(arg330_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg331_1, (1024, ), (1, ))
    assert_size_stride(arg332_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg333_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg334_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg335_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg336_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg337_1, (16, 64), (64, 1))
    assert_size_stride(arg338_1, (16, 64), (64, 1))
    assert_size_stride(arg339_1, (1024, ), (1, ))
    assert_size_stride(arg340_1, (1024, ), (1, ))
    assert_size_stride(arg341_1, (1024, ), (1, ))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg344_1, (4096, ), (1, ))
    assert_size_stride(arg345_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg346_1, (1024, ), (1, ))
    assert_size_stride(arg347_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg348_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg349_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg350_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg351_1, (1024, 16, 64), (1024, 64, 1))
    assert_size_stride(arg352_1, (16, 64), (64, 1))
    assert_size_stride(arg353_1, (16, 64), (64, 1))
    assert_size_stride(arg354_1, (1024, ), (1, ))
    assert_size_stride(arg355_1, (1024, ), (1, ))
    assert_size_stride(arg356_1, (1024, ), (1, ))
    assert_size_stride(arg357_1, (1024, ), (1, ))
    assert_size_stride(arg358_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg359_1, (4096, ), (1, ))
    assert_size_stride(arg360_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (32000, ), (1, ))
    assert_size_stride(arg363_1, (8, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 8, 1024), (8192, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_ids, word_emb_k], Original ATen: [aten.clone, aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_embedding_0.run(arg0_1, arg1_1, buf0, 4194304, grid=grid(4194304), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 4096, 1024), (4194304, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(arg2_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf1)
        del arg2_1
        buf2 = empty_strided_cuda((1, 4096, 1024), (4194304, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg3_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf2)
        del arg3_1
        buf3 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        buf11 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf1, arg8_1, arg7_1, buf3, buf11, 4194304, grid=grid(4194304), stream=stream0)
        del arg7_1
        del arg8_1
        buf4 = empty_strided_cuda((128, 512, 512), (262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ac], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf2, (128, 64, 512), (64, 1, 8192), 0), out=buf4)
    buf7 = empty_strided_cpu((1024, 1024), (1024, 1), torch.float32)
    buf5 = reinterpret_tensor(buf7, (1024, 512), (1024, 1), 0)  # alias
    buf6 = reinterpret_tensor(buf7, (1024, 512), (1024, 1), 512)  # alias
    cpp_fused_cos_mul_sin_2(buf5, buf6)
    del buf5
    del buf6
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf8 = empty_strided_cuda((1024, 8, 1024), (8192, 1024, 1), torch.float32)
        buf8.copy_(reinterpret_tensor(buf7, (1024, 8, 1024), (1024, 0, 1), 0))
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [pos_emb_3], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf9, 8388608, grid=grid(8388608), stream=stream0)
        buf10 = empty_strided_cuda((1, 8192, 1024), (8388608, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k_head_r], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (0, 1024, 1), 0), reinterpret_tensor(arg6_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf10)
        del arg6_1
        buf12 = empty_strided_cuda((128, 512, 1024), (524288, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bd], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf10, (128, 64, 1024), (64, 1, 8192), 0), out=buf12)
        buf16 = empty_strided_cuda((8, 16, 512, 512), (4194304, 262144, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, add_2, add_3, attn_prob], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf4, buf12, buf16, 65536, 512, grid=grid(65536), stream=stream0)
        buf15 = reinterpret_tensor(buf11, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [v_head_h], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg4_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf15)
        del arg4_1
        buf17 = reinterpret_tensor(buf3, (128, 512, 64), (32768, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [attn_vec], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf15, (128, 512, 64), (64, 8192, 1), 0), out=buf17)
        buf18 = reinterpret_tensor(buf15, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf17, buf18, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf19 = empty_strided_cuda((64, 16, 1024, 1, 1), (16384, 1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg5_1, buf19, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg5_1
        buf20 = reinterpret_tensor(buf17, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [attn_out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf19, (1, 1024, 1024), (0, 1024, 1), 0), out=buf20)
        buf24 = reinterpret_tensor(buf18, (512, 8, 1024), (8192, 1024, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [attn_out_2, output], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf20, buf0, arg9_1, arg10_1, buf24, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg10_1
        del arg9_1
        buf25 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf24, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 4096), (1, 1024), 0), out=buf25)
        del arg13_1
        buf26 = reinterpret_tensor(buf25, (512, 8, 4096), (32768, 4096, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [output_2], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf26, arg14_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg14_1
        buf27 = reinterpret_tensor(buf20, (4096, 1024), (1024, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf26, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg15_1, (4096, 1024), (1, 4096), 0), out=buf27)
        del arg15_1
        buf31 = reinterpret_tensor(buf2, (512, 8, 1024), (8192, 1024, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [add_5, output_6], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf27, arg16_1, buf24, arg11_1, arg12_1, buf31, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg11_1
        del arg12_1
        del arg16_1
        buf32 = reinterpret_tensor(buf27, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg17_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf32)
        del arg17_1
        buf33 = reinterpret_tensor(buf24, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg18_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf33)
        del arg18_1
        buf34 = reinterpret_tensor(buf1, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf1  # reuse
        buf37 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_6, add_7], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf32, arg23_1, arg22_1, buf34, buf37, 4194304, grid=grid(4194304), stream=stream0)
        del arg22_1
        del arg23_1
        buf35 = reinterpret_tensor(buf16, (128, 512, 512), (262144, 512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [ac_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf33, (128, 64, 512), (64, 1, 8192), 0), out=buf35)
        buf36 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg21_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf36)
        del arg21_1
        buf38 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [bd_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf36, (128, 64, 1024), (64, 1, 8192), 0), out=buf38)
        buf42 = reinterpret_tensor(buf4, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_7, add_8, add_9, attn_prob_2], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf35, buf38, buf42, 65536, 512, grid=grid(65536), stream=stream0)
        buf41 = reinterpret_tensor(buf37, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf31, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg19_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf41)
        del arg19_1
        buf43 = reinterpret_tensor(buf34, (128, 512, 64), (32768, 64, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf41, (128, 512, 64), (64, 8192, 1), 0), out=buf43)
        buf44 = reinterpret_tensor(buf41, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf43, buf44, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf45 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [attn_out_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg20_1, buf45, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg20_1
        buf46 = reinterpret_tensor(buf43, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [attn_out_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf45, (1, 1024, 1024), (0, 1024, 1), 0), out=buf46)
        buf50 = reinterpret_tensor(buf44, (512, 8, 1024), (8192, 1024, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [attn_out_5, output_7], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf46, buf31, arg24_1, arg25_1, buf50, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg24_1
        del arg25_1
        buf51 = reinterpret_tensor(buf26, (4096, 4096), (4096, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf50, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 4096), (1, 1024), 0), out=buf51)
        del arg28_1
        buf52 = reinterpret_tensor(buf51, (512, 8, 4096), (32768, 4096, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [output_9], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf52, arg29_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg29_1
        buf53 = reinterpret_tensor(buf46, (4096, 1024), (1024, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg30_1, (4096, 1024), (1, 4096), 0), out=buf53)
        del arg30_1
        buf57 = reinterpret_tensor(buf33, (512, 8, 1024), (8192, 1024, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [add_11, output_13], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf53, arg31_1, buf50, arg26_1, arg27_1, buf57, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg26_1
        del arg27_1
        del arg31_1
        buf58 = reinterpret_tensor(buf53, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg32_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf58)
        del arg32_1
        buf59 = reinterpret_tensor(buf50, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg33_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf59)
        del arg33_1
        buf60 = reinterpret_tensor(buf32, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf32  # reuse
        buf63 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_12, add_13], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf58, arg38_1, arg37_1, buf60, buf63, 4194304, grid=grid(4194304), stream=stream0)
        del arg37_1
        del arg38_1
        buf61 = reinterpret_tensor(buf42, (128, 512, 512), (262144, 512, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [ac_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf59, (128, 64, 512), (64, 1, 8192), 0), out=buf61)
        buf62 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg36_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf62)
        del arg36_1
        buf64 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [bd_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf62, (128, 64, 1024), (64, 1, 8192), 0), out=buf64)
        buf68 = reinterpret_tensor(buf35, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_11, add_14, add_15, attn_prob_4], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf61, buf64, buf68, 65536, 512, grid=grid(65536), stream=stream0)
        buf67 = reinterpret_tensor(buf63, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg34_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf67)
        del arg34_1
        buf69 = reinterpret_tensor(buf60, (128, 512, 64), (32768, 64, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf67, (128, 512, 64), (64, 8192, 1), 0), out=buf69)
        buf70 = reinterpret_tensor(buf67, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf69, buf70, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf71 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [attn_out_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg35_1, buf71, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg35_1
        buf72 = reinterpret_tensor(buf69, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [attn_out_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf71, (1, 1024, 1024), (0, 1024, 1), 0), out=buf72)
        buf76 = reinterpret_tensor(buf70, (512, 8, 1024), (8192, 1024, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [attn_out_8, output_14], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf72, buf57, arg39_1, arg40_1, buf76, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg39_1
        del arg40_1
        buf77 = reinterpret_tensor(buf52, (4096, 4096), (4096, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf76, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg43_1, (1024, 4096), (1, 1024), 0), out=buf77)
        del arg43_1
        buf78 = reinterpret_tensor(buf77, (512, 8, 4096), (32768, 4096, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [output_16], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf78, arg44_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg44_1
        buf79 = reinterpret_tensor(buf72, (4096, 1024), (1024, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf78, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg45_1, (4096, 1024), (1, 4096), 0), out=buf79)
        del arg45_1
        buf83 = reinterpret_tensor(buf59, (512, 8, 1024), (8192, 1024, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [add_17, output_20], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf79, arg46_1, buf76, arg41_1, arg42_1, buf83, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg41_1
        del arg42_1
        del arg46_1
        buf84 = reinterpret_tensor(buf79, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg47_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf84)
        del arg47_1
        buf85 = reinterpret_tensor(buf76, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg48_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf85)
        del arg48_1
        buf86 = reinterpret_tensor(buf58, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf58  # reuse
        buf89 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_18, add_19], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf84, arg53_1, arg52_1, buf86, buf89, 4194304, grid=grid(4194304), stream=stream0)
        del arg52_1
        del arg53_1
        buf87 = reinterpret_tensor(buf68, (128, 512, 512), (262144, 512, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [ac_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf86, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf85, (128, 64, 512), (64, 1, 8192), 0), out=buf87)
        buf88 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg51_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf88)
        del arg51_1
        buf90 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [bd_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf88, (128, 64, 1024), (64, 1, 8192), 0), out=buf90)
        buf94 = reinterpret_tensor(buf61, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_15, add_20, add_21, attn_prob_6], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf87, buf90, buf94, 65536, 512, grid=grid(65536), stream=stream0)
        buf93 = reinterpret_tensor(buf89, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg49_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf93)
        del arg49_1
        buf95 = reinterpret_tensor(buf86, (128, 512, 64), (32768, 64, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf93, (128, 512, 64), (64, 8192, 1), 0), out=buf95)
        buf96 = reinterpret_tensor(buf93, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf95, buf96, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf97 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [attn_out_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg50_1, buf97, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg50_1
        buf98 = reinterpret_tensor(buf95, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [attn_out_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf97, (1, 1024, 1024), (0, 1024, 1), 0), out=buf98)
        buf102 = reinterpret_tensor(buf96, (512, 8, 1024), (8192, 1024, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [attn_out_11, output_21], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf98, buf83, arg54_1, arg55_1, buf102, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg54_1
        del arg55_1
        buf103 = reinterpret_tensor(buf78, (4096, 4096), (4096, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg58_1, (1024, 4096), (1, 1024), 0), out=buf103)
        del arg58_1
        buf104 = reinterpret_tensor(buf103, (512, 8, 4096), (32768, 4096, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [output_23], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf104, arg59_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg59_1
        buf105 = reinterpret_tensor(buf98, (4096, 1024), (1024, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf104, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg60_1, (4096, 1024), (1, 4096), 0), out=buf105)
        del arg60_1
        buf109 = reinterpret_tensor(buf85, (512, 8, 1024), (8192, 1024, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [add_23, output_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf105, arg61_1, buf102, arg56_1, arg57_1, buf109, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg56_1
        del arg57_1
        del arg61_1
        buf110 = reinterpret_tensor(buf105, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg62_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf110)
        del arg62_1
        buf111 = reinterpret_tensor(buf102, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg63_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf111)
        del arg63_1
        buf112 = reinterpret_tensor(buf84, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf84  # reuse
        buf115 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_24, add_25], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf110, arg68_1, arg67_1, buf112, buf115, 4194304, grid=grid(4194304), stream=stream0)
        del arg67_1
        del arg68_1
        buf113 = reinterpret_tensor(buf94, (128, 512, 512), (262144, 512, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [ac_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf112, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf111, (128, 64, 512), (64, 1, 8192), 0), out=buf113)
        buf114 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg66_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf114)
        del arg66_1
        buf116 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [bd_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf114, (128, 64, 1024), (64, 1, 8192), 0), out=buf116)
        buf120 = reinterpret_tensor(buf87, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_19, add_26, add_27, attn_prob_8], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf113, buf116, buf120, 65536, 512, grid=grid(65536), stream=stream0)
        buf119 = reinterpret_tensor(buf115, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg64_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf119)
        del arg64_1
        buf121 = reinterpret_tensor(buf112, (128, 512, 64), (32768, 64, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf119, (128, 512, 64), (64, 8192, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf119, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf121, buf122, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf123 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [attn_out_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg65_1, buf123, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg65_1
        buf124 = reinterpret_tensor(buf121, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [attn_out_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf123, (1, 1024, 1024), (0, 1024, 1), 0), out=buf124)
        buf128 = reinterpret_tensor(buf122, (512, 8, 1024), (8192, 1024, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [attn_out_14, output_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf124, buf109, arg69_1, arg70_1, buf128, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg69_1
        del arg70_1
        buf129 = reinterpret_tensor(buf104, (4096, 4096), (4096, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 4096), (1, 1024), 0), out=buf129)
        del arg73_1
        buf130 = reinterpret_tensor(buf129, (512, 8, 4096), (32768, 4096, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [output_30], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf130, arg74_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg74_1
        buf131 = reinterpret_tensor(buf124, (4096, 1024), (1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf130, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg75_1, (4096, 1024), (1, 4096), 0), out=buf131)
        del arg75_1
        buf135 = reinterpret_tensor(buf111, (512, 8, 1024), (8192, 1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [add_29, output_34], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf131, arg76_1, buf128, arg71_1, arg72_1, buf135, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg71_1
        del arg72_1
        del arg76_1
        buf136 = reinterpret_tensor(buf131, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg77_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf136)
        del arg77_1
        buf137 = reinterpret_tensor(buf128, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg78_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf137)
        del arg78_1
        buf138 = reinterpret_tensor(buf110, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf110  # reuse
        buf141 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_30, add_31], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf136, arg83_1, arg82_1, buf138, buf141, 4194304, grid=grid(4194304), stream=stream0)
        del arg82_1
        del arg83_1
        buf139 = reinterpret_tensor(buf120, (128, 512, 512), (262144, 512, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [ac_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf138, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf137, (128, 64, 512), (64, 1, 8192), 0), out=buf139)
        buf140 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg81_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf140)
        del arg81_1
        buf142 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [bd_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf140, (128, 64, 1024), (64, 1, 8192), 0), out=buf142)
        buf146 = reinterpret_tensor(buf113, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_23, add_32, add_33, attn_prob_10], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf139, buf142, buf146, 65536, 512, grid=grid(65536), stream=stream0)
        buf145 = reinterpret_tensor(buf141, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg79_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf145)
        del arg79_1
        buf147 = reinterpret_tensor(buf138, (128, 512, 64), (32768, 64, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf145, (128, 512, 64), (64, 8192, 1), 0), out=buf147)
        buf148 = reinterpret_tensor(buf145, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf147, buf148, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf149 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [attn_out_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg80_1, buf149, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg80_1
        buf150 = reinterpret_tensor(buf147, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [attn_out_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf149, (1, 1024, 1024), (0, 1024, 1), 0), out=buf150)
        buf154 = reinterpret_tensor(buf148, (512, 8, 1024), (8192, 1024, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [attn_out_17, output_35], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf150, buf135, arg84_1, arg85_1, buf154, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg84_1
        del arg85_1
        buf155 = reinterpret_tensor(buf130, (4096, 4096), (4096, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf154, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg88_1, (1024, 4096), (1, 1024), 0), out=buf155)
        del arg88_1
        buf156 = reinterpret_tensor(buf155, (512, 8, 4096), (32768, 4096, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [output_37], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf156, arg89_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg89_1
        buf157 = reinterpret_tensor(buf150, (4096, 1024), (1024, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf156, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg90_1, (4096, 1024), (1, 4096), 0), out=buf157)
        del arg90_1
        buf161 = reinterpret_tensor(buf137, (512, 8, 1024), (8192, 1024, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [add_35, output_41], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf157, arg91_1, buf154, arg86_1, arg87_1, buf161, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg86_1
        del arg87_1
        del arg91_1
        buf162 = reinterpret_tensor(buf157, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg92_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf162)
        del arg92_1
        buf163 = reinterpret_tensor(buf154, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg93_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf163)
        del arg93_1
        buf164 = reinterpret_tensor(buf136, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf136  # reuse
        buf167 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_36, add_37], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf162, arg98_1, arg97_1, buf164, buf167, 4194304, grid=grid(4194304), stream=stream0)
        del arg97_1
        del arg98_1
        buf165 = reinterpret_tensor(buf146, (128, 512, 512), (262144, 512, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [ac_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf163, (128, 64, 512), (64, 1, 8192), 0), out=buf165)
        buf166 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg96_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf166)
        del arg96_1
        buf168 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [bd_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf166, (128, 64, 1024), (64, 1, 8192), 0), out=buf168)
        buf172 = reinterpret_tensor(buf139, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_27, add_38, add_39, attn_prob_12], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf165, buf168, buf172, 65536, 512, grid=grid(65536), stream=stream0)
        buf171 = reinterpret_tensor(buf167, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf161, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg94_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf171)
        del arg94_1
        buf173 = reinterpret_tensor(buf164, (128, 512, 64), (32768, 64, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf171, (128, 512, 64), (64, 8192, 1), 0), out=buf173)
        buf174 = reinterpret_tensor(buf171, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf173, buf174, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf175 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [attn_out_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg95_1, buf175, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg95_1
        buf176 = reinterpret_tensor(buf173, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [attn_out_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf174, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf175, (1, 1024, 1024), (0, 1024, 1), 0), out=buf176)
        buf180 = reinterpret_tensor(buf174, (512, 8, 1024), (8192, 1024, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [attn_out_20, output_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf176, buf161, arg99_1, arg100_1, buf180, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg100_1
        del arg99_1
        buf181 = reinterpret_tensor(buf156, (4096, 4096), (4096, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf180, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg103_1, (1024, 4096), (1, 1024), 0), out=buf181)
        del arg103_1
        buf182 = reinterpret_tensor(buf181, (512, 8, 4096), (32768, 4096, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf182, arg104_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg104_1
        buf183 = reinterpret_tensor(buf176, (4096, 1024), (1024, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf182, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg105_1, (4096, 1024), (1, 4096), 0), out=buf183)
        del arg105_1
        buf187 = reinterpret_tensor(buf163, (512, 8, 1024), (8192, 1024, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [add_41, output_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf183, arg106_1, buf180, arg101_1, arg102_1, buf187, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg101_1
        del arg102_1
        del arg106_1
        buf188 = reinterpret_tensor(buf183, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg107_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf188)
        del arg107_1
        buf189 = reinterpret_tensor(buf180, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg108_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf189)
        del arg108_1
        buf190 = reinterpret_tensor(buf162, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf162  # reuse
        buf193 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_42, add_43], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf188, arg113_1, arg112_1, buf190, buf193, 4194304, grid=grid(4194304), stream=stream0)
        del arg112_1
        del arg113_1
        buf191 = reinterpret_tensor(buf172, (128, 512, 512), (262144, 512, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [ac_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf190, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf189, (128, 64, 512), (64, 1, 8192), 0), out=buf191)
        buf192 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg111_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf192)
        del arg111_1
        buf194 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [bd_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf192, (128, 64, 1024), (64, 1, 8192), 0), out=buf194)
        buf198 = reinterpret_tensor(buf165, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_31, add_44, add_45, attn_prob_14], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf191, buf194, buf198, 65536, 512, grid=grid(65536), stream=stream0)
        buf197 = reinterpret_tensor(buf193, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg109_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf197)
        del arg109_1
        buf199 = reinterpret_tensor(buf190, (128, 512, 64), (32768, 64, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf197, (128, 512, 64), (64, 8192, 1), 0), out=buf199)
        buf200 = reinterpret_tensor(buf197, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf199, buf200, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf201 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [attn_out_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg110_1, buf201, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg110_1
        buf202 = reinterpret_tensor(buf199, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [attn_out_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf200, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf201, (1, 1024, 1024), (0, 1024, 1), 0), out=buf202)
        buf206 = reinterpret_tensor(buf200, (512, 8, 1024), (8192, 1024, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [attn_out_23, output_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf202, buf187, arg114_1, arg115_1, buf206, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg114_1
        del arg115_1
        buf207 = reinterpret_tensor(buf182, (4096, 4096), (4096, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf206, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg118_1, (1024, 4096), (1, 1024), 0), out=buf207)
        del arg118_1
        buf208 = reinterpret_tensor(buf207, (512, 8, 4096), (32768, 4096, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [output_51], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf208, arg119_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg119_1
        buf209 = reinterpret_tensor(buf202, (4096, 1024), (1024, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf208, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg120_1, (4096, 1024), (1, 4096), 0), out=buf209)
        del arg120_1
        buf213 = reinterpret_tensor(buf189, (512, 8, 1024), (8192, 1024, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [add_47, output_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf209, arg121_1, buf206, arg116_1, arg117_1, buf213, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg116_1
        del arg117_1
        del arg121_1
        buf214 = reinterpret_tensor(buf209, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg122_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf214)
        del arg122_1
        buf215 = reinterpret_tensor(buf206, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg123_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf215)
        del arg123_1
        buf216 = reinterpret_tensor(buf188, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf188  # reuse
        buf219 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_48, add_49], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf214, arg128_1, arg127_1, buf216, buf219, 4194304, grid=grid(4194304), stream=stream0)
        del arg127_1
        del arg128_1
        buf217 = reinterpret_tensor(buf198, (128, 512, 512), (262144, 512, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [ac_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf215, (128, 64, 512), (64, 1, 8192), 0), out=buf217)
        buf218 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg126_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf218)
        del arg126_1
        buf220 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [bd_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf219, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf218, (128, 64, 1024), (64, 1, 8192), 0), out=buf220)
        buf224 = reinterpret_tensor(buf191, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_35, add_50, add_51, attn_prob_16], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf217, buf220, buf224, 65536, 512, grid=grid(65536), stream=stream0)
        buf223 = reinterpret_tensor(buf219, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg124_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf223)
        del arg124_1
        buf225 = reinterpret_tensor(buf216, (128, 512, 64), (32768, 64, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf223, (128, 512, 64), (64, 8192, 1), 0), out=buf225)
        buf226 = reinterpret_tensor(buf223, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf225, buf226, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf227 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [attn_out_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg125_1, buf227, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg125_1
        buf228 = reinterpret_tensor(buf225, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [attn_out_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf227, (1, 1024, 1024), (0, 1024, 1), 0), out=buf228)
        buf232 = reinterpret_tensor(buf226, (512, 8, 1024), (8192, 1024, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [attn_out_26, output_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf228, buf213, arg129_1, arg130_1, buf232, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg129_1
        del arg130_1
        buf233 = reinterpret_tensor(buf208, (4096, 4096), (4096, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf232, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 4096), (1, 1024), 0), out=buf233)
        del arg133_1
        buf234 = reinterpret_tensor(buf233, (512, 8, 4096), (32768, 4096, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [output_58], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf234, arg134_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg134_1
        buf235 = reinterpret_tensor(buf228, (4096, 1024), (1024, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf234, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg135_1, (4096, 1024), (1, 4096), 0), out=buf235)
        del arg135_1
        buf239 = reinterpret_tensor(buf215, (512, 8, 1024), (8192, 1024, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [add_53, output_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf235, arg136_1, buf232, arg131_1, arg132_1, buf239, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg131_1
        del arg132_1
        del arg136_1
        buf240 = reinterpret_tensor(buf235, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg137_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf240)
        del arg137_1
        buf241 = reinterpret_tensor(buf232, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg138_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf241)
        del arg138_1
        buf242 = reinterpret_tensor(buf214, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf214  # reuse
        buf245 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_54, add_55], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf240, arg143_1, arg142_1, buf242, buf245, 4194304, grid=grid(4194304), stream=stream0)
        del arg142_1
        del arg143_1
        buf243 = reinterpret_tensor(buf224, (128, 512, 512), (262144, 512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [ac_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf241, (128, 64, 512), (64, 1, 8192), 0), out=buf243)
        buf244 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg141_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf244)
        del arg141_1
        buf246 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [bd_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf245, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf244, (128, 64, 1024), (64, 1, 8192), 0), out=buf246)
        buf250 = reinterpret_tensor(buf217, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_39, add_56, add_57, attn_prob_18], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf243, buf246, buf250, 65536, 512, grid=grid(65536), stream=stream0)
        buf249 = reinterpret_tensor(buf245, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg139_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf249)
        del arg139_1
        buf251 = reinterpret_tensor(buf242, (128, 512, 64), (32768, 64, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf249, (128, 512, 64), (64, 8192, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf249, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf251, buf252, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf253 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [attn_out_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg140_1, buf253, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg140_1
        buf254 = reinterpret_tensor(buf251, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [attn_out_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf253, (1, 1024, 1024), (0, 1024, 1), 0), out=buf254)
        buf258 = reinterpret_tensor(buf252, (512, 8, 1024), (8192, 1024, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [attn_out_29, output_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf254, buf239, arg144_1, arg145_1, buf258, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg144_1
        del arg145_1
        buf259 = reinterpret_tensor(buf234, (4096, 4096), (4096, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 4096), (1, 1024), 0), out=buf259)
        del arg148_1
        buf260 = reinterpret_tensor(buf259, (512, 8, 4096), (32768, 4096, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [output_65], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf260, arg149_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg149_1
        buf261 = reinterpret_tensor(buf254, (4096, 1024), (1024, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf260, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg150_1, (4096, 1024), (1, 4096), 0), out=buf261)
        del arg150_1
        buf265 = reinterpret_tensor(buf241, (512, 8, 1024), (8192, 1024, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [add_59, output_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf261, arg151_1, buf258, arg146_1, arg147_1, buf265, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg146_1
        del arg147_1
        del arg151_1
        buf266 = reinterpret_tensor(buf261, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg152_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf266)
        del arg152_1
        buf267 = reinterpret_tensor(buf258, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg153_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf267)
        del arg153_1
        buf268 = reinterpret_tensor(buf240, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf240  # reuse
        buf271 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_60, add_61], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf266, arg158_1, arg157_1, buf268, buf271, 4194304, grid=grid(4194304), stream=stream0)
        del arg157_1
        del arg158_1
        buf269 = reinterpret_tensor(buf250, (128, 512, 512), (262144, 512, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [ac_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf267, (128, 64, 512), (64, 1, 8192), 0), out=buf269)
        buf270 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg156_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf270)
        del arg156_1
        buf272 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [bd_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf271, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf270, (128, 64, 1024), (64, 1, 8192), 0), out=buf272)
        buf276 = reinterpret_tensor(buf243, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_43, add_62, add_63, attn_prob_20], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf269, buf272, buf276, 65536, 512, grid=grid(65536), stream=stream0)
        buf275 = reinterpret_tensor(buf271, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg154_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf275)
        del arg154_1
        buf277 = reinterpret_tensor(buf268, (128, 512, 64), (32768, 64, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf275, (128, 512, 64), (64, 8192, 1), 0), out=buf277)
        buf278 = reinterpret_tensor(buf275, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf277, buf278, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf279 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [attn_out_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg155_1, buf279, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg155_1
        buf280 = reinterpret_tensor(buf277, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [attn_out_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf279, (1, 1024, 1024), (0, 1024, 1), 0), out=buf280)
        buf284 = reinterpret_tensor(buf278, (512, 8, 1024), (8192, 1024, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [attn_out_32, output_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf280, buf265, arg159_1, arg160_1, buf284, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg159_1
        del arg160_1
        buf285 = reinterpret_tensor(buf260, (4096, 4096), (4096, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf284, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg163_1, (1024, 4096), (1, 1024), 0), out=buf285)
        del arg163_1
        buf286 = reinterpret_tensor(buf285, (512, 8, 4096), (32768, 4096, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [output_72], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf286, arg164_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg164_1
        buf287 = reinterpret_tensor(buf280, (4096, 1024), (1024, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg165_1, (4096, 1024), (1, 4096), 0), out=buf287)
        del arg165_1
        buf291 = reinterpret_tensor(buf267, (512, 8, 1024), (8192, 1024, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [add_65, output_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf287, arg166_1, buf284, arg161_1, arg162_1, buf291, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg161_1
        del arg162_1
        del arg166_1
        buf292 = reinterpret_tensor(buf287, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg167_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf292)
        del arg167_1
        buf293 = reinterpret_tensor(buf284, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg168_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf293)
        del arg168_1
        buf294 = reinterpret_tensor(buf266, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf266  # reuse
        buf297 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_66, add_67], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf292, arg173_1, arg172_1, buf294, buf297, 4194304, grid=grid(4194304), stream=stream0)
        del arg172_1
        del arg173_1
        buf295 = reinterpret_tensor(buf276, (128, 512, 512), (262144, 512, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [ac_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf293, (128, 64, 512), (64, 1, 8192), 0), out=buf295)
        buf296 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg171_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf296)
        del arg171_1
        buf298 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [bd_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf297, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf296, (128, 64, 1024), (64, 1, 8192), 0), out=buf298)
        buf302 = reinterpret_tensor(buf269, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_47, add_68, add_69, attn_prob_22], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf295, buf298, buf302, 65536, 512, grid=grid(65536), stream=stream0)
        buf301 = reinterpret_tensor(buf297, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg169_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf301)
        del arg169_1
        buf303 = reinterpret_tensor(buf294, (128, 512, 64), (32768, 64, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf301, (128, 512, 64), (64, 8192, 1), 0), out=buf303)
        buf304 = reinterpret_tensor(buf301, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf303, buf304, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf305 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [attn_out_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg170_1, buf305, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg170_1
        buf306 = reinterpret_tensor(buf303, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [attn_out_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf305, (1, 1024, 1024), (0, 1024, 1), 0), out=buf306)
        buf310 = reinterpret_tensor(buf304, (512, 8, 1024), (8192, 1024, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [attn_out_35, output_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf306, buf291, arg174_1, arg175_1, buf310, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg174_1
        del arg175_1
        buf311 = reinterpret_tensor(buf286, (4096, 4096), (4096, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf310, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg178_1, (1024, 4096), (1, 1024), 0), out=buf311)
        del arg178_1
        buf312 = reinterpret_tensor(buf311, (512, 8, 4096), (32768, 4096, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [output_79], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf312, arg179_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg179_1
        buf313 = reinterpret_tensor(buf306, (4096, 1024), (1024, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf312, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg180_1, (4096, 1024), (1, 4096), 0), out=buf313)
        del arg180_1
        buf317 = reinterpret_tensor(buf293, (512, 8, 1024), (8192, 1024, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [add_71, output_83], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf313, arg181_1, buf310, arg176_1, arg177_1, buf317, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg176_1
        del arg177_1
        del arg181_1
        buf318 = reinterpret_tensor(buf313, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg182_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf318)
        del arg182_1
        buf319 = reinterpret_tensor(buf310, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg183_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf319)
        del arg183_1
        buf320 = reinterpret_tensor(buf292, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf292  # reuse
        buf323 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_72, add_73], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf318, arg188_1, arg187_1, buf320, buf323, 4194304, grid=grid(4194304), stream=stream0)
        del arg187_1
        del arg188_1
        buf321 = reinterpret_tensor(buf302, (128, 512, 512), (262144, 512, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [ac_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf319, (128, 64, 512), (64, 1, 8192), 0), out=buf321)
        buf322 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg186_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf322)
        del arg186_1
        buf324 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [bd_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf323, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf322, (128, 64, 1024), (64, 1, 8192), 0), out=buf324)
        buf328 = reinterpret_tensor(buf295, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_51, add_74, add_75, attn_prob_24], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf321, buf324, buf328, 65536, 512, grid=grid(65536), stream=stream0)
        buf327 = reinterpret_tensor(buf323, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf317, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg184_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf327)
        del arg184_1
        buf329 = reinterpret_tensor(buf320, (128, 512, 64), (32768, 64, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf327, (128, 512, 64), (64, 8192, 1), 0), out=buf329)
        buf330 = reinterpret_tensor(buf327, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf329, buf330, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf331 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [attn_out_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg185_1, buf331, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg185_1
        buf332 = reinterpret_tensor(buf329, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [attn_out_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf331, (1, 1024, 1024), (0, 1024, 1), 0), out=buf332)
        buf336 = reinterpret_tensor(buf330, (512, 8, 1024), (8192, 1024, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [attn_out_38, output_84], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf332, buf317, arg189_1, arg190_1, buf336, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg189_1
        del arg190_1
        buf337 = reinterpret_tensor(buf312, (4096, 4096), (4096, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 4096), (1, 1024), 0), out=buf337)
        del arg193_1
        buf338 = reinterpret_tensor(buf337, (512, 8, 4096), (32768, 4096, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [output_86], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf338, arg194_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg194_1
        buf339 = reinterpret_tensor(buf332, (4096, 1024), (1024, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf338, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 1024), (1, 4096), 0), out=buf339)
        del arg195_1
        buf343 = reinterpret_tensor(buf319, (512, 8, 1024), (8192, 1024, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [add_77, output_90], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf339, arg196_1, buf336, arg191_1, arg192_1, buf343, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg191_1
        del arg192_1
        del arg196_1
        buf344 = reinterpret_tensor(buf339, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg197_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf344)
        del arg197_1
        buf345 = reinterpret_tensor(buf336, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg198_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf345)
        del arg198_1
        buf346 = reinterpret_tensor(buf318, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf318  # reuse
        buf349 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_78, add_79], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf344, arg203_1, arg202_1, buf346, buf349, 4194304, grid=grid(4194304), stream=stream0)
        del arg202_1
        del arg203_1
        buf347 = reinterpret_tensor(buf328, (128, 512, 512), (262144, 512, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [ac_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf346, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf345, (128, 64, 512), (64, 1, 8192), 0), out=buf347)
        buf348 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg201_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf348)
        del arg201_1
        buf350 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [bd_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf349, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf348, (128, 64, 1024), (64, 1, 8192), 0), out=buf350)
        buf354 = reinterpret_tensor(buf321, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_55, add_80, add_81, attn_prob_26], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf347, buf350, buf354, 65536, 512, grid=grid(65536), stream=stream0)
        buf353 = reinterpret_tensor(buf349, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf343, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg199_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf353)
        del arg199_1
        buf355 = reinterpret_tensor(buf346, (128, 512, 64), (32768, 64, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf353, (128, 512, 64), (64, 8192, 1), 0), out=buf355)
        buf356 = reinterpret_tensor(buf353, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf355, buf356, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf357 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [attn_out_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg200_1, buf357, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg200_1
        buf358 = reinterpret_tensor(buf355, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [attn_out_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf357, (1, 1024, 1024), (0, 1024, 1), 0), out=buf358)
        buf362 = reinterpret_tensor(buf356, (512, 8, 1024), (8192, 1024, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [attn_out_41, output_91], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf358, buf343, arg204_1, arg205_1, buf362, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg204_1
        del arg205_1
        buf363 = reinterpret_tensor(buf338, (4096, 4096), (4096, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf362, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg208_1, (1024, 4096), (1, 1024), 0), out=buf363)
        del arg208_1
        buf364 = reinterpret_tensor(buf363, (512, 8, 4096), (32768, 4096, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [output_93], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf364, arg209_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg209_1
        buf365 = reinterpret_tensor(buf358, (4096, 1024), (1024, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg210_1, (4096, 1024), (1, 4096), 0), out=buf365)
        del arg210_1
        buf369 = reinterpret_tensor(buf345, (512, 8, 1024), (8192, 1024, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [add_83, output_97], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf365, arg211_1, buf362, arg206_1, arg207_1, buf369, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg206_1
        del arg207_1
        del arg211_1
        buf370 = reinterpret_tensor(buf365, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg212_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf370)
        del arg212_1
        buf371 = reinterpret_tensor(buf362, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg213_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf371)
        del arg213_1
        buf372 = reinterpret_tensor(buf344, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf344  # reuse
        buf375 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_84, add_85], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf370, arg218_1, arg217_1, buf372, buf375, 4194304, grid=grid(4194304), stream=stream0)
        del arg217_1
        del arg218_1
        buf373 = reinterpret_tensor(buf354, (128, 512, 512), (262144, 512, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [ac_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf371, (128, 64, 512), (64, 1, 8192), 0), out=buf373)
        buf374 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg216_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf374)
        del arg216_1
        buf376 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [bd_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf375, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf374, (128, 64, 1024), (64, 1, 8192), 0), out=buf376)
        buf380 = reinterpret_tensor(buf347, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf347  # reuse
        # Topologically Sorted Source Nodes: [x_59, add_86, add_87, attn_prob_28], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf373, buf376, buf380, 65536, 512, grid=grid(65536), stream=stream0)
        buf379 = reinterpret_tensor(buf375, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg214_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf379)
        del arg214_1
        buf381 = reinterpret_tensor(buf372, (128, 512, 64), (32768, 64, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf379, (128, 512, 64), (64, 8192, 1), 0), out=buf381)
        buf382 = reinterpret_tensor(buf379, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf381, buf382, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf383 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [attn_out_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg215_1, buf383, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg215_1
        buf384 = reinterpret_tensor(buf381, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [attn_out_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf382, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf383, (1, 1024, 1024), (0, 1024, 1), 0), out=buf384)
        buf388 = reinterpret_tensor(buf382, (512, 8, 1024), (8192, 1024, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [attn_out_44, output_98], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf384, buf369, arg219_1, arg220_1, buf388, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg219_1
        del arg220_1
        buf389 = reinterpret_tensor(buf364, (4096, 4096), (4096, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf388, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg223_1, (1024, 4096), (1, 1024), 0), out=buf389)
        del arg223_1
        buf390 = reinterpret_tensor(buf389, (512, 8, 4096), (32768, 4096, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [output_100], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf390, arg224_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg224_1
        buf391 = reinterpret_tensor(buf384, (4096, 1024), (1024, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf390, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg225_1, (4096, 1024), (1, 4096), 0), out=buf391)
        del arg225_1
        buf395 = reinterpret_tensor(buf371, (512, 8, 1024), (8192, 1024, 1), 0); del buf371  # reuse
        # Topologically Sorted Source Nodes: [add_89, output_104], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf391, arg226_1, buf388, arg221_1, arg222_1, buf395, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg221_1
        del arg222_1
        del arg226_1
        buf396 = reinterpret_tensor(buf391, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg227_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf396)
        del arg227_1
        buf397 = reinterpret_tensor(buf388, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg228_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf397)
        del arg228_1
        buf398 = reinterpret_tensor(buf370, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf370  # reuse
        buf401 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_90, add_91], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf396, arg233_1, arg232_1, buf398, buf401, 4194304, grid=grid(4194304), stream=stream0)
        del arg232_1
        del arg233_1
        buf399 = reinterpret_tensor(buf380, (128, 512, 512), (262144, 512, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [ac_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf398, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf397, (128, 64, 512), (64, 1, 8192), 0), out=buf399)
        buf400 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg231_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf400)
        del arg231_1
        buf402 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [bd_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf401, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf400, (128, 64, 1024), (64, 1, 8192), 0), out=buf402)
        buf406 = reinterpret_tensor(buf373, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [x_63, add_92, add_93, attn_prob_30], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf399, buf402, buf406, 65536, 512, grid=grid(65536), stream=stream0)
        buf405 = reinterpret_tensor(buf401, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf401  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg229_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf405)
        del arg229_1
        buf407 = reinterpret_tensor(buf398, (128, 512, 64), (32768, 64, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf405, (128, 512, 64), (64, 8192, 1), 0), out=buf407)
        buf408 = reinterpret_tensor(buf405, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf407, buf408, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf409 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [attn_out_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg230_1, buf409, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg230_1
        buf410 = reinterpret_tensor(buf407, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf407  # reuse
        # Topologically Sorted Source Nodes: [attn_out_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf409, (1, 1024, 1024), (0, 1024, 1), 0), out=buf410)
        buf414 = reinterpret_tensor(buf408, (512, 8, 1024), (8192, 1024, 1), 0); del buf408  # reuse
        # Topologically Sorted Source Nodes: [attn_out_47, output_105], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf410, buf395, arg234_1, arg235_1, buf414, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg234_1
        del arg235_1
        buf415 = reinterpret_tensor(buf390, (4096, 4096), (4096, 1), 0); del buf390  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf414, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg238_1, (1024, 4096), (1, 1024), 0), out=buf415)
        del arg238_1
        buf416 = reinterpret_tensor(buf415, (512, 8, 4096), (32768, 4096, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [output_107], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf416, arg239_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg239_1
        buf417 = reinterpret_tensor(buf410, (4096, 1024), (1024, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf416, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg240_1, (4096, 1024), (1, 4096), 0), out=buf417)
        del arg240_1
        buf421 = reinterpret_tensor(buf397, (512, 8, 1024), (8192, 1024, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [add_95, output_111], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf417, arg241_1, buf414, arg236_1, arg237_1, buf421, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg236_1
        del arg237_1
        del arg241_1
        buf422 = reinterpret_tensor(buf417, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg242_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf422)
        del arg242_1
        buf423 = reinterpret_tensor(buf414, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg243_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf423)
        del arg243_1
        buf424 = reinterpret_tensor(buf396, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf396  # reuse
        buf427 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_96, add_97], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf422, arg248_1, arg247_1, buf424, buf427, 4194304, grid=grid(4194304), stream=stream0)
        del arg247_1
        del arg248_1
        buf425 = reinterpret_tensor(buf406, (128, 512, 512), (262144, 512, 1), 0); del buf406  # reuse
        # Topologically Sorted Source Nodes: [ac_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf424, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf423, (128, 64, 512), (64, 1, 8192), 0), out=buf425)
        buf426 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg246_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf426)
        del arg246_1
        buf428 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [bd_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf427, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf426, (128, 64, 1024), (64, 1, 8192), 0), out=buf428)
        buf432 = reinterpret_tensor(buf399, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_67, add_98, add_99, attn_prob_32], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf425, buf428, buf432, 65536, 512, grid=grid(65536), stream=stream0)
        buf431 = reinterpret_tensor(buf427, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg244_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf431)
        del arg244_1
        buf433 = reinterpret_tensor(buf424, (128, 512, 64), (32768, 64, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf431, (128, 512, 64), (64, 8192, 1), 0), out=buf433)
        buf434 = reinterpret_tensor(buf431, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf433, buf434, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf435 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [attn_out_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg245_1, buf435, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg245_1
        buf436 = reinterpret_tensor(buf433, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf433  # reuse
        # Topologically Sorted Source Nodes: [attn_out_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf434, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf435, (1, 1024, 1024), (0, 1024, 1), 0), out=buf436)
        buf440 = reinterpret_tensor(buf434, (512, 8, 1024), (8192, 1024, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [attn_out_50, output_112], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf436, buf421, arg249_1, arg250_1, buf440, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg249_1
        del arg250_1
        buf441 = reinterpret_tensor(buf416, (4096, 4096), (4096, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg253_1, (1024, 4096), (1, 1024), 0), out=buf441)
        del arg253_1
        buf442 = reinterpret_tensor(buf441, (512, 8, 4096), (32768, 4096, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [output_114], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf442, arg254_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg254_1
        buf443 = reinterpret_tensor(buf436, (4096, 1024), (1024, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf442, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg255_1, (4096, 1024), (1, 4096), 0), out=buf443)
        del arg255_1
        buf447 = reinterpret_tensor(buf423, (512, 8, 1024), (8192, 1024, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [add_101, output_118], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf443, arg256_1, buf440, arg251_1, arg252_1, buf447, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg251_1
        del arg252_1
        del arg256_1
        buf448 = reinterpret_tensor(buf443, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg257_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf448)
        del arg257_1
        buf449 = reinterpret_tensor(buf440, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg258_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf449)
        del arg258_1
        buf450 = reinterpret_tensor(buf422, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf422  # reuse
        buf453 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_102, add_103], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf448, arg263_1, arg262_1, buf450, buf453, 4194304, grid=grid(4194304), stream=stream0)
        del arg262_1
        del arg263_1
        buf451 = reinterpret_tensor(buf432, (128, 512, 512), (262144, 512, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [ac_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf450, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf449, (128, 64, 512), (64, 1, 8192), 0), out=buf451)
        buf452 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg261_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf452)
        del arg261_1
        buf454 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [bd_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf453, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf452, (128, 64, 1024), (64, 1, 8192), 0), out=buf454)
        buf458 = reinterpret_tensor(buf425, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [x_71, add_104, add_105, attn_prob_34], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf451, buf454, buf458, 65536, 512, grid=grid(65536), stream=stream0)
        buf457 = reinterpret_tensor(buf453, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf447, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg259_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf457)
        del arg259_1
        buf459 = reinterpret_tensor(buf450, (128, 512, 64), (32768, 64, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf458, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf457, (128, 512, 64), (64, 8192, 1), 0), out=buf459)
        buf460 = reinterpret_tensor(buf457, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf457  # reuse
        # Topologically Sorted Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf459, buf460, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf461 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [attn_out_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg260_1, buf461, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg260_1
        buf462 = reinterpret_tensor(buf459, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [attn_out_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf460, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf461, (1, 1024, 1024), (0, 1024, 1), 0), out=buf462)
        buf466 = reinterpret_tensor(buf460, (512, 8, 1024), (8192, 1024, 1), 0); del buf460  # reuse
        # Topologically Sorted Source Nodes: [attn_out_53, output_119], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf462, buf447, arg264_1, arg265_1, buf466, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg264_1
        del arg265_1
        buf467 = reinterpret_tensor(buf442, (4096, 4096), (4096, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf466, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg268_1, (1024, 4096), (1, 1024), 0), out=buf467)
        del arg268_1
        buf468 = reinterpret_tensor(buf467, (512, 8, 4096), (32768, 4096, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [output_121], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf468, arg269_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg269_1
        buf469 = reinterpret_tensor(buf462, (4096, 1024), (1024, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf468, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg270_1, (4096, 1024), (1, 4096), 0), out=buf469)
        del arg270_1
        buf473 = reinterpret_tensor(buf449, (512, 8, 1024), (8192, 1024, 1), 0); del buf449  # reuse
        # Topologically Sorted Source Nodes: [add_107, output_125], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf469, arg271_1, buf466, arg266_1, arg267_1, buf473, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg266_1
        del arg267_1
        del arg271_1
        buf474 = reinterpret_tensor(buf469, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg272_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf474)
        del arg272_1
        buf475 = reinterpret_tensor(buf466, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf466  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg273_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf475)
        del arg273_1
        buf476 = reinterpret_tensor(buf448, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf448  # reuse
        buf479 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_108, add_109], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf474, arg278_1, arg277_1, buf476, buf479, 4194304, grid=grid(4194304), stream=stream0)
        del arg277_1
        del arg278_1
        buf477 = reinterpret_tensor(buf458, (128, 512, 512), (262144, 512, 1), 0); del buf458  # reuse
        # Topologically Sorted Source Nodes: [ac_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf476, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf475, (128, 64, 512), (64, 1, 8192), 0), out=buf477)
        buf478 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg276_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf478)
        del arg276_1
        buf480 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [bd_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf479, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf478, (128, 64, 1024), (64, 1, 8192), 0), out=buf480)
        buf484 = reinterpret_tensor(buf451, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_75, add_110, add_111, attn_prob_36], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf477, buf480, buf484, 65536, 512, grid=grid(65536), stream=stream0)
        buf483 = reinterpret_tensor(buf479, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf479  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf473, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg274_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf483)
        del arg274_1
        buf485 = reinterpret_tensor(buf476, (128, 512, 64), (32768, 64, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf484, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf483, (128, 512, 64), (64, 8192, 1), 0), out=buf485)
        buf486 = reinterpret_tensor(buf483, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf483  # reuse
        # Topologically Sorted Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf485, buf486, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf487 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [attn_out_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg275_1, buf487, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg275_1
        buf488 = reinterpret_tensor(buf485, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf485  # reuse
        # Topologically Sorted Source Nodes: [attn_out_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf487, (1, 1024, 1024), (0, 1024, 1), 0), out=buf488)
        buf492 = reinterpret_tensor(buf486, (512, 8, 1024), (8192, 1024, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [attn_out_56, output_126], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf488, buf473, arg279_1, arg280_1, buf492, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg279_1
        del arg280_1
        buf493 = reinterpret_tensor(buf468, (4096, 4096), (4096, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf492, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg283_1, (1024, 4096), (1, 1024), 0), out=buf493)
        del arg283_1
        buf494 = reinterpret_tensor(buf493, (512, 8, 4096), (32768, 4096, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [output_128], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf494, arg284_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg284_1
        buf495 = reinterpret_tensor(buf488, (4096, 1024), (1024, 1), 0); del buf488  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf494, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg285_1, (4096, 1024), (1, 4096), 0), out=buf495)
        del arg285_1
        buf499 = reinterpret_tensor(buf475, (512, 8, 1024), (8192, 1024, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [add_113, output_132], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf495, arg286_1, buf492, arg281_1, arg282_1, buf499, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg281_1
        del arg282_1
        del arg286_1
        buf500 = reinterpret_tensor(buf495, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg287_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf500)
        del arg287_1
        buf501 = reinterpret_tensor(buf492, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf492  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg288_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf501)
        del arg288_1
        buf502 = reinterpret_tensor(buf474, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf474  # reuse
        buf505 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_114, add_115], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf500, arg293_1, arg292_1, buf502, buf505, 4194304, grid=grid(4194304), stream=stream0)
        del arg292_1
        del arg293_1
        buf503 = reinterpret_tensor(buf484, (128, 512, 512), (262144, 512, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [ac_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf501, (128, 64, 512), (64, 1, 8192), 0), out=buf503)
        buf504 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg291_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf504)
        del arg291_1
        buf506 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [bd_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf505, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf504, (128, 64, 1024), (64, 1, 8192), 0), out=buf506)
        buf510 = reinterpret_tensor(buf477, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf477  # reuse
        # Topologically Sorted Source Nodes: [x_79, add_116, add_117, attn_prob_38], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf503, buf506, buf510, 65536, 512, grid=grid(65536), stream=stream0)
        buf509 = reinterpret_tensor(buf505, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg289_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf509)
        del arg289_1
        buf511 = reinterpret_tensor(buf502, (128, 512, 64), (32768, 64, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf510, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf509, (128, 512, 64), (64, 8192, 1), 0), out=buf511)
        buf512 = reinterpret_tensor(buf509, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf509  # reuse
        # Topologically Sorted Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf511, buf512, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf513 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [attn_out_57], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg290_1, buf513, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg290_1
        buf514 = reinterpret_tensor(buf511, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [attn_out_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf512, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf513, (1, 1024, 1024), (0, 1024, 1), 0), out=buf514)
        buf518 = reinterpret_tensor(buf512, (512, 8, 1024), (8192, 1024, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [attn_out_59, output_133], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf514, buf499, arg294_1, arg295_1, buf518, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg294_1
        del arg295_1
        buf519 = reinterpret_tensor(buf494, (4096, 4096), (4096, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf518, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg298_1, (1024, 4096), (1, 1024), 0), out=buf519)
        del arg298_1
        buf520 = reinterpret_tensor(buf519, (512, 8, 4096), (32768, 4096, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [output_135], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf520, arg299_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg299_1
        buf521 = reinterpret_tensor(buf514, (4096, 1024), (1024, 1), 0); del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf520, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg300_1, (4096, 1024), (1, 4096), 0), out=buf521)
        del arg300_1
        buf525 = reinterpret_tensor(buf501, (512, 8, 1024), (8192, 1024, 1), 0); del buf501  # reuse
        # Topologically Sorted Source Nodes: [add_119, output_139], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf521, arg301_1, buf518, arg296_1, arg297_1, buf525, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg296_1
        del arg297_1
        del arg301_1
        buf526 = reinterpret_tensor(buf521, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg302_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf526)
        del arg302_1
        buf527 = reinterpret_tensor(buf518, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg303_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf527)
        del arg303_1
        buf528 = reinterpret_tensor(buf500, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf500  # reuse
        buf531 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_120, add_121], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf526, arg308_1, arg307_1, buf528, buf531, 4194304, grid=grid(4194304), stream=stream0)
        del arg307_1
        del arg308_1
        buf529 = reinterpret_tensor(buf510, (128, 512, 512), (262144, 512, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [ac_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf528, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf527, (128, 64, 512), (64, 1, 8192), 0), out=buf529)
        buf530 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg306_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf530)
        del arg306_1
        buf532 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [bd_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf531, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf530, (128, 64, 1024), (64, 1, 8192), 0), out=buf532)
        buf536 = reinterpret_tensor(buf503, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf503  # reuse
        # Topologically Sorted Source Nodes: [x_83, add_122, add_123, attn_prob_40], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf529, buf532, buf536, 65536, 512, grid=grid(65536), stream=stream0)
        buf535 = reinterpret_tensor(buf531, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf531  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf525, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg304_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf535)
        del arg304_1
        buf537 = reinterpret_tensor(buf528, (128, 512, 64), (32768, 64, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf536, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf535, (128, 512, 64), (64, 8192, 1), 0), out=buf537)
        buf538 = reinterpret_tensor(buf535, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf535  # reuse
        # Topologically Sorted Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf537, buf538, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf539 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [attn_out_60], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg305_1, buf539, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg305_1
        buf540 = reinterpret_tensor(buf537, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [attn_out_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf538, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf539, (1, 1024, 1024), (0, 1024, 1), 0), out=buf540)
        buf544 = reinterpret_tensor(buf538, (512, 8, 1024), (8192, 1024, 1), 0); del buf538  # reuse
        # Topologically Sorted Source Nodes: [attn_out_62, output_140], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf540, buf525, arg309_1, arg310_1, buf544, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg309_1
        del arg310_1
        buf545 = reinterpret_tensor(buf520, (4096, 4096), (4096, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf544, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg313_1, (1024, 4096), (1, 1024), 0), out=buf545)
        del arg313_1
        buf546 = reinterpret_tensor(buf545, (512, 8, 4096), (32768, 4096, 1), 0); del buf545  # reuse
        # Topologically Sorted Source Nodes: [output_142], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf546, arg314_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg314_1
        buf547 = reinterpret_tensor(buf540, (4096, 1024), (1024, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg315_1, (4096, 1024), (1, 4096), 0), out=buf547)
        del arg315_1
        buf551 = reinterpret_tensor(buf527, (512, 8, 1024), (8192, 1024, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [add_125, output_146], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf547, arg316_1, buf544, arg311_1, arg312_1, buf551, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg311_1
        del arg312_1
        del arg316_1
        buf552 = reinterpret_tensor(buf547, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf551, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg317_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf552)
        del arg317_1
        buf553 = reinterpret_tensor(buf544, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf544  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf551, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg318_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf553)
        del arg318_1
        buf554 = reinterpret_tensor(buf526, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf526  # reuse
        buf557 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_126, add_127], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf552, arg323_1, arg322_1, buf554, buf557, 4194304, grid=grid(4194304), stream=stream0)
        del arg322_1
        del arg323_1
        buf555 = reinterpret_tensor(buf536, (128, 512, 512), (262144, 512, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [ac_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf554, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf553, (128, 64, 512), (64, 1, 8192), 0), out=buf555)
        buf556 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg321_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf556)
        del arg321_1
        buf558 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [bd_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf556, (128, 64, 1024), (64, 1, 8192), 0), out=buf558)
        buf562 = reinterpret_tensor(buf529, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [x_87, add_128, add_129, attn_prob_42], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf555, buf558, buf562, 65536, 512, grid=grid(65536), stream=stream0)
        buf561 = reinterpret_tensor(buf557, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf551, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg319_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf561)
        del arg319_1
        buf563 = reinterpret_tensor(buf554, (128, 512, 64), (32768, 64, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf562, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf561, (128, 512, 64), (64, 8192, 1), 0), out=buf563)
        buf564 = reinterpret_tensor(buf561, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf561  # reuse
        # Topologically Sorted Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf563, buf564, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf565 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [attn_out_63], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg320_1, buf565, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg320_1
        buf566 = reinterpret_tensor(buf563, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf563  # reuse
        # Topologically Sorted Source Nodes: [attn_out_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf564, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf565, (1, 1024, 1024), (0, 1024, 1), 0), out=buf566)
        buf570 = reinterpret_tensor(buf564, (512, 8, 1024), (8192, 1024, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [attn_out_65, output_147], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf566, buf551, arg324_1, arg325_1, buf570, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg324_1
        del arg325_1
        buf571 = reinterpret_tensor(buf546, (4096, 4096), (4096, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf570, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg328_1, (1024, 4096), (1, 1024), 0), out=buf571)
        del arg328_1
        buf572 = reinterpret_tensor(buf571, (512, 8, 4096), (32768, 4096, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [output_149], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf572, arg329_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg329_1
        buf573 = reinterpret_tensor(buf566, (4096, 1024), (1024, 1), 0); del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf572, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg330_1, (4096, 1024), (1, 4096), 0), out=buf573)
        del arg330_1
        buf577 = reinterpret_tensor(buf553, (512, 8, 1024), (8192, 1024, 1), 0); del buf553  # reuse
        # Topologically Sorted Source Nodes: [add_131, output_153], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf573, arg331_1, buf570, arg326_1, arg327_1, buf577, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg326_1
        del arg327_1
        del arg331_1
        buf578 = reinterpret_tensor(buf573, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf573  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg332_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf578)
        del arg332_1
        buf579 = reinterpret_tensor(buf570, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg333_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf579)
        del arg333_1
        buf580 = reinterpret_tensor(buf552, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf552  # reuse
        buf583 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_132, add_133], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf578, arg338_1, arg337_1, buf580, buf583, 4194304, grid=grid(4194304), stream=stream0)
        del arg337_1
        del arg338_1
        buf581 = reinterpret_tensor(buf562, (128, 512, 512), (262144, 512, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [ac_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf580, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf579, (128, 64, 512), (64, 1, 8192), 0), out=buf581)
        buf582 = buf556; del buf556  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg336_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf582)
        del arg336_1
        buf584 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [bd_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf583, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf582, (128, 64, 1024), (64, 1, 8192), 0), out=buf584)
        buf588 = reinterpret_tensor(buf555, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [x_91, add_134, add_135, attn_prob_44], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf581, buf584, buf588, 65536, 512, grid=grid(65536), stream=stream0)
        buf587 = reinterpret_tensor(buf583, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg334_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf587)
        del arg334_1
        buf589 = reinterpret_tensor(buf580, (128, 512, 64), (32768, 64, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf587, (128, 512, 64), (64, 8192, 1), 0), out=buf589)
        buf590 = reinterpret_tensor(buf587, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf587  # reuse
        # Topologically Sorted Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf589, buf590, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf591 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [attn_out_66], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg335_1, buf591, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg335_1
        buf592 = reinterpret_tensor(buf589, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf589  # reuse
        # Topologically Sorted Source Nodes: [attn_out_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf590, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf591, (1, 1024, 1024), (0, 1024, 1), 0), out=buf592)
        buf596 = reinterpret_tensor(buf590, (512, 8, 1024), (8192, 1024, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [attn_out_68, output_154], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf592, buf577, arg339_1, arg340_1, buf596, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg339_1
        del arg340_1
        buf597 = reinterpret_tensor(buf572, (4096, 4096), (4096, 1), 0); del buf572  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf596, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg343_1, (1024, 4096), (1, 1024), 0), out=buf597)
        del arg343_1
        buf598 = reinterpret_tensor(buf597, (512, 8, 4096), (32768, 4096, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [output_156], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf598, arg344_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg344_1
        buf599 = reinterpret_tensor(buf592, (4096, 1024), (1024, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf598, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg345_1, (4096, 1024), (1, 4096), 0), out=buf599)
        del arg345_1
        buf603 = reinterpret_tensor(buf579, (512, 8, 1024), (8192, 1024, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [add_137, output_160], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf599, arg346_1, buf596, arg341_1, arg342_1, buf603, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg341_1
        del arg342_1
        del arg346_1
        buf604 = reinterpret_tensor(buf599, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [q_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf603, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg347_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf604)
        del arg347_1
        buf605 = reinterpret_tensor(buf596, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf596  # reuse
        # Topologically Sorted Source Nodes: [k_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf603, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg348_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf605)
        del arg348_1
        buf606 = reinterpret_tensor(buf578, (512, 8, 16, 64), (8192, 1024, 64, 1), 0); del buf578  # reuse
        buf609 = empty_strided_cuda((512, 8, 16, 64), (8192, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_138, add_139], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf604, arg353_1, arg352_1, buf606, buf609, 4194304, grid=grid(4194304), stream=stream0)
        del arg352_1
        del arg353_1
        del buf604
        buf607 = reinterpret_tensor(buf588, (128, 512, 512), (262144, 512, 1), 0); del buf588  # reuse
        # Topologically Sorted Source Nodes: [ac_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf606, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf605, (128, 64, 512), (64, 1, 8192), 0), out=buf607)
        buf608 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [k_head_r_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (1, 8192, 1024), (8388608, 1024, 1), 0), reinterpret_tensor(arg351_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf608)
        del arg351_1
        del buf9
        buf610 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [bd_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf609, (128, 512, 64), (64, 8192, 1), 0), reinterpret_tensor(buf608, (128, 64, 1024), (64, 1, 8192), 0), out=buf610)
        del buf608
        buf614 = reinterpret_tensor(buf581, (8, 16, 512, 512), (4194304, 262144, 512, 1), 0); del buf581  # reuse
        # Topologically Sorted Source Nodes: [x_95, add_140, add_141, attn_prob_46], Original ATen: [aten.index_select, aten.add, aten._softmax]
        triton_red_fused__softmax_add_index_select_4.run(buf607, buf610, buf614, 65536, 512, grid=grid(65536), stream=stream0)
        del buf607
        del buf610
        buf613 = reinterpret_tensor(buf609, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf609  # reuse
        # Topologically Sorted Source Nodes: [v_head_h_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf603, (1, 4096, 1024), (4194304, 1024, 1), 0), reinterpret_tensor(arg349_1, (1, 1024, 1024), (1048576, 1024, 1), 0), out=buf613)
        del arg349_1
        buf615 = reinterpret_tensor(buf606, (128, 512, 64), (32768, 64, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [attn_vec_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf614, (128, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf613, (128, 512, 64), (64, 8192, 1), 0), out=buf615)
        del buf614
        buf616 = reinterpret_tensor(buf613, (512, 8, 64, 16, 1), (8192, 1024, 16, 1, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf615, buf616, 262144, 16, grid=grid(262144, 16), stream=stream0)
        buf617 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [attn_out_69], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(arg350_1, buf617, 64, 16384, grid=grid(64, 16384), stream=stream0)
        del arg350_1
        buf618 = reinterpret_tensor(buf615, (1, 4096, 1024), (4194304, 1024, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [attn_out_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf616, (1, 4096, 1024), (0, 1024, 1), 0), reinterpret_tensor(buf617, (1, 1024, 1024), (0, 1024, 1), 0), out=buf618)
        del buf617
        buf622 = reinterpret_tensor(buf616, (512, 8, 1024), (8192, 1024, 1), 0); del buf616  # reuse
        # Topologically Sorted Source Nodes: [attn_out_71, output_161], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_7.run(buf618, buf603, arg354_1, arg355_1, buf622, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg354_1
        del arg355_1
        buf623 = reinterpret_tensor(buf598, (4096, 4096), (4096, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf622, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg358_1, (1024, 4096), (1, 1024), 0), out=buf623)
        del arg358_1
        buf624 = reinterpret_tensor(buf623, (512, 8, 4096), (32768, 4096, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [output_163], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_8.run(buf624, arg359_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg359_1
        buf625 = reinterpret_tensor(buf618, (4096, 1024), (1024, 1), 0); del buf618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf624, (4096, 4096), (4096, 1), 0), reinterpret_tensor(arg360_1, (4096, 1024), (1, 4096), 0), out=buf625)
        del arg360_1
        del buf624
        buf629 = reinterpret_tensor(buf605, (8, 512, 1024), (524288, 1024, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [add_143, output_167, output_169], Original ATen: [aten.add, aten.native_layer_norm, aten.clone]
        triton_per_fused_add_clone_native_layer_norm_10.run(buf625, arg361_1, buf622, arg356_1, arg357_1, buf629, 4096, 1024, grid=grid(4096), stream=stream0)
        del arg356_1
        del arg357_1
        del arg361_1
        del buf622
        del buf625
        buf630 = empty_strided_cuda((4096, 32000), (32000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg362_1, reinterpret_tensor(buf629, (4096, 1024), (1024, 1), 0), reinterpret_tensor(arg1_1, (1024, 32000), (1, 1024), 0), alpha=1, beta=1, out=buf630)
        del arg1_1
        del arg362_1
        del buf629
        buf631 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        buf632 = empty_strided_cuda((4096, 1), (1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_11.run(buf630, buf631, buf632, 4096, 32000, grid=grid(4096), stream=stream0)
        buf633 = empty_strided_cuda((), (), torch.float32)
        buf635 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_12.run(buf635, arg363_1, buf630, buf631, buf632, 1, 4096, grid=grid(1), stream=stream0)
        del arg363_1
        del buf631
        del buf632
    return (buf635, reinterpret_tensor(buf630, (8, 512, 32000), (16384000, 32000, 1), 0), buf0, buf31, buf57, buf83, buf109, buf135, buf161, buf187, buf213, buf239, buf265, buf291, buf317, buf343, buf369, buf395, buf421, buf447, buf473, buf499, buf525, buf551, buf577, buf603, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((32000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((8, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XLNetLMHeadModel', benchmark_compiled_module)
