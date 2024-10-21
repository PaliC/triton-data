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


# kernel path: /tmp/torchinductor_sahanp/eh/ceht27vbsmlt6ox5m7xhqumvk5yvhnv6w4p3xfjvfagojzfql7rr.py
# Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_215 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (65536*y3)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (196608*y1)), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lt/cltiu3szd73oc7uccpazylbbmynvcroh24nhc4luunkmkbitxhos.py
# Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_215 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2k/c2kcnhj7at6rjf3oqoclufgp35ckrra7eoj3wv37eg6nccyyxfo3.py
# Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_216 => add_127, mul_173, mul_174, sub_53
#   x_217 => mul_175, sigmoid_34
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_257), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_259), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_173, %unsqueeze_261), kwargs = {})
#   %add_127 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_174, %unsqueeze_263), kwargs = {})
#   %sigmoid_34 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_127,), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_127, %sigmoid_34), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ac/cacf4z5bzklk5q2auykfcph3sekj35vc2dpe76dg5tkgzdjeigss.py
# Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_219 => add_129, mul_177, mul_178, sub_54
#   x_220 => mul_179, sigmoid_35
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_265), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_267), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_177, %unsqueeze_269), kwargs = {})
#   %add_129 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_178, %unsqueeze_271), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_129,), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_129, %sigmoid_35), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/62/c623oxgwfmorxl7flqt2in72mlb6n4cjiuxdj4u544on6rjddg7e.py
# Topologically Sorted Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_225 => add_133, mul_185, mul_186, sub_56
# Graph fragment:
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_281), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_283), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_185, %unsqueeze_285), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, %unsqueeze_287), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/l3/cl3b3pgfpccka7baypquw3z7s76v5woq6pdkd6nw5zsobcktxnah.py
# Topologically Sorted Source Nodes: [x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_227 => add_135, mul_188, mul_189, sub_57
#   x_228 => mul_190, sigmoid_37
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_289), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_291), kwargs = {})
#   %mul_189 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_188, %unsqueeze_293), kwargs = {})
#   %add_135 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_189, %unsqueeze_295), kwargs = {})
#   %sigmoid_37 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_135,), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_135, %sigmoid_37), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d2/cd2wqrs7zoj5ricbbvhek2dio5mnrjsruovdmq2v6jz2jybo3em7.py
# Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_230 => add_137, mul_192, mul_193, sub_58
#   x_231 => mul_194, sigmoid_38
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_297), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_299), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_192, %unsqueeze_301), kwargs = {})
#   %add_137 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_193, %unsqueeze_303), kwargs = {})
#   %sigmoid_38 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_137,), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, %sigmoid_38), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ww/cwwkv65kzxladoefceogh4ha3lnb63oepwq5gcrirywtak5xykfy.py
# Topologically Sorted Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_233 => add_139, mul_196, mul_197, sub_59
# Graph fragment:
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_305), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %unsqueeze_307), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_309), kwargs = {})
#   %add_139 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_311), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g3/cg3hvs6dg67qsbgcaokoxz3mr6xhydj5wdcsnlau6o4a2inkzm5g.py
# Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_235 => add_141, mul_199, mul_200, sub_60
#   x_236 => mul_201, sigmoid_39
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_313), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_315), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_317), kwargs = {})
#   %add_141 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_319), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_141,), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_141, %sigmoid_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r6/cr6p5dp3yfy545joapmubbo73utks5265ztlhfy5fekoc2shuwjq.py
# Topologically Sorted Source Nodes: [x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_241 => add_145, mul_207, mul_208, sub_62
#   x_242 => add_146
# Graph fragment:
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_329), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_331), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_207, %unsqueeze_333), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_208, %unsqueeze_335), kwargs = {})
#   %add_146 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_145, %add_139), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mw/cmwkle467qaqntk7xwcehkklpcq46gctfxvzyhuwb3bmajrvf3yr.py
# Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_256 => add_157, mul_225, mul_226, sub_67
#   x_257 => mul_227, sigmoid_44
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_369), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_371), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_225, %unsqueeze_373), kwargs = {})
#   %add_157 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_226, %unsqueeze_375), kwargs = {})
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_157,), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_157, %sigmoid_44), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d3/cd3dby3dvnrybpfwk2jy4thmjgkstfod3jblcytcpbpmcyyvyiih.py
# Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_259 => add_159, mul_229, mul_230, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_377), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_379), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_229, %unsqueeze_381), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_230, %unsqueeze_383), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ie/cievcrggdugdvbcqtpdc6tfqovmcgzs4xi4moqnksgefipdxtvpf.py
# Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_260 => convolution_51
# Graph fragment:
#   %convolution_51 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_159, %arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uo/cuom7eygtuoljmdmwuloab6l5enov6tizml3xlc7vdjp7nsnbc24.py
# Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_261 => add_161, mul_232, mul_233, sub_69
#   x_262 => mul_234, sigmoid_45
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_385), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_387), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_232, %unsqueeze_389), kwargs = {})
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_233, %unsqueeze_391), kwargs = {})
#   %sigmoid_45 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_161,), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %sigmoid_45), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4d/c4d2jeauvnlmgqbaj3dj7mjovrkq4t5rrefywmqwqn7zy74npvse.py
# Topologically Sorted Source Nodes: [layer_norm_21], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_21 => add_162, add_163, mul_235, mul_236, rsqrt_21, sub_70, var_mean_21
# Graph fragment:
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_111, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_111, %getitem_106), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_105, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_162,), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %rsqrt_21), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_235, %arg87_1), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, %arg88_1), kwargs = {})
triton_per_fused_native_layer_norm_14 = async_compile.triton('triton_per_fused_native_layer_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_14(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (144*((x0 % 4) % 2)) + (288*((((4*x1) + (x0 % 4)) // 4) % 16)) + (4608*(((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + ((x0 % 4) % 2)) // 32)) + (4608*((x0 % 4) // 2)) + (9216*(((4*x1) + (x0 % 4)) // 64)) + (147456*(x0 // 4)) + (147456*(triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*(((4*x1) + (x0 % 4)) // 64)) + (1024*r2) + ((x0 % 4) % 2),  147456))) + (triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*(((4*x1) + (x0 % 4)) // 64)) + ((x0 % 4) % 2),  1024))), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 144.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (144*x1) + (36864*x0)), tmp27, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ks/cksmuwo6gx2z2xmpeszzecagbqyhelfaud2ckir5hm7cogpem5g4.py
# Topologically Sorted Source Nodes: [x_270, layer_norm_22], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_22 => add_165, add_166, mul_237, mul_238, rsqrt_22, sub_71, var_mean_22
#   x_270 => add_164
# Graph fragment:
#   %add_164 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_111, %view_117), kwargs = {})
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_164, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_164, %getitem_115), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_114, 1e-05), kwargs = {})
#   %rsqrt_22 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_165,), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %rsqrt_22), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %arg93_1), kwargs = {})
#   %add_166 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %arg94_1), kwargs = {})
triton_per_fused_add_native_layer_norm_15 = async_compile.triton('triton_per_fused_add_native_layer_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (144*((x1 % 4) % 2)) + (288*((((4*x0) + (x1 % 4)) // 4) % 16)) + (4608*(((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + ((x1 % 4) % 2)) // 32)) + (4608*((x1 % 4) // 2)) + (9216*(((4*x0) + (x1 % 4)) // 64)) + (147456*(x1 // 4)) + (147456*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*(((4*x0) + (x1 % 4)) // 64)) + (1024*r2) + ((x1 % 4) % 2),  147456))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*(((4*x0) + (x1 % 4)) // 64)) + ((x1 % 4) % 2),  1024))), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 144.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (144*x3)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4zlyticyuogeqiu64pnmozvs5fbtndnyaa74xyc33mftkz5yw6.py
# Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   x_272 => mul_239, sigmoid_46
# Graph fragment:
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_119,), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_119, %sigmoid_46), kwargs = {})
triton_poi_fused_silu_16 = async_compile.triton('triton_poi_fused_silu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_16(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 288
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6c7y52lt3wzd7mkhyulk4esnvve4gi7iqcwizyu5odz6ikxsr6e.py
# Topologically Sorted Source Nodes: [x_270, x_276, layer_norm_23], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_23 => add_168, add_169, mul_240, mul_241, rsqrt_23, sub_72, var_mean_23
#   x_270 => add_164
#   x_276 => add_167
# Graph fragment:
#   %add_164 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_111, %view_117), kwargs = {})
#   %add_167 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_164, %view_121), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_167, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_167, %getitem_117), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_116, 1e-05), kwargs = {})
#   %rsqrt_23 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_168,), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %rsqrt_23), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_240, %arg99_1), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_241, %arg100_1), kwargs = {})
triton_per_fused_add_native_layer_norm_17 = async_compile.triton('triton_per_fused_add_native_layer_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (144*((x1 % 4) % 2)) + (288*((((4*x0) + (x1 % 4)) // 4) % 16)) + (4608*(((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + ((x1 % 4) % 2)) // 32)) + (4608*((x1 % 4) // 2)) + (9216*(((4*x0) + (x1 % 4)) // 64)) + (147456*(x1 // 4)) + (147456*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*(((4*x0) + (x1 % 4)) // 64)) + (1024*r2) + ((x1 % 4) % 2),  147456))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*(((4*x0) + (x1 % 4)) // 64)) + ((x1 % 4) % 2),  1024))), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 144.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp8, rmask)
    tl.store(out_ptr2 + (r2 + (144*x3)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rv/crvaufnhjti57ehacss25ba6jorvfnalwertrly56jxra2cmv2xr.py
# Topologically Sorted Source Nodes: [x_281, layer_norm_24], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_24 => add_171, add_172, mul_242, mul_243, rsqrt_24, sub_73, var_mean_24
#   x_281 => add_170
# Graph fragment:
#   %add_170 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %view_127), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_170, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_170, %getitem_126), kwargs = {})
#   %add_171 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_125, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_171,), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %rsqrt_24), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %arg105_1), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %arg106_1), kwargs = {})
triton_per_fused_add_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 144.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dy/cdyocucqy4za5npewj2vwku622nxezhksktbpkovfaplsc4zbqyf.py
# Topologically Sorted Source Nodes: [x_281, x_287, x_288], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_281 => add_170
#   x_287 => add_173
#   x_288 => var_mean_25
# Graph fragment:
#   %add_170 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %view_127), kwargs = {})
#   %add_173 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %view_131), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_173, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_19 = async_compile.triton('triton_per_fused_add_native_layer_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp8, rmask)
    tl.store(out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4n3xfkuad4hrhk67kqj34ng74xxvlod7k4i3lxnlqmrwqjzflz.py
# Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_291 => clone_49
# Graph fragment:
#   %clone_49 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_82,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4, 524288], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 294912
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + ((144*(((y0 + (2*y1) + (4*x2) + (64*x3)) // 4) % 256)) + (36864*y0) + (73728*y1) + (147456*((y0 + (2*y1) + (4*x2) + (64*x3)) // 147456)) + (((y0 + (2*y1) + (4*x2) + (64*x3)) // 1024) % 144)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((256*y0) + (512*y1) + (1024*((y0 + (2*y1) + (4*x2) + (64*x3)) // 147456)) + (((y0 + (2*y1) + (4*x2) + (64*x3)) // 4) % 256)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((256*y0) + (512*y1) + (1024*((y0 + (2*y1) + (4*x2) + (64*x3)) // 147456)) + (((y0 + (2*y1) + (4*x2) + (64*x3)) // 4) % 256)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (((y0 + (2*y1) + (4*x2) + (64*x3)) // 1024) % 144), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (((y0 + (2*y1) + (4*x2) + (64*x3)) // 1024) % 144), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 144.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (2*x2) + (32*y1) + (64*x3)), tmp13, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6d/c6didktzzfrkjn4ksdrns2qa4zoxdl25dvcypbzpcpmzvrggqajc.py
# Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_292 => convolution_53
# Graph fragment:
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_134, %arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 144
    y1 = (yindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*(x3 % 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y4)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (144*x5) + (147456*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o6/co6ocoasabbdzpu7fxekjjj6f4mjskqo72dutxkhbo2ufqtnavi3.py
# Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_159, %mul_250], 1), kwargs = {})
triton_poi_fused_cat_22 = async_compile.triton('triton_poi_fused_cat_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((96*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((96*x1) + ((-96) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g4/cg4nfah4gvpu43hotznuwd3xi6pg4gdgley62pz5ymls3pegmwys.py
# Topologically Sorted Source Nodes: [cat_3, x_295], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   cat_3 => cat_3
#   x_295 => convolution_54
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_159, %mul_250], 1), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg118_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_23 = async_compile.triton('triton_poi_fused_cat_convolution_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rw/crwv6fndhglma6wneogrxwxyubka72b77auqmqsicfwbkscfy4nb.py
# Topologically Sorted Source Nodes: [x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_299 => add_181, mul_256, mul_257, sub_77
#   x_300 => mul_258, sigmoid_50
# Graph fragment:
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_409), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_411), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_413), kwargs = {})
#   %add_181 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_415), kwargs = {})
#   %sigmoid_50 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_181,), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_181, %sigmoid_50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ee/ceesxokhmoem4446qcfkud4qknqwgpp5cwwfia2regyblyalcqsb.py
# Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_302 => add_183, mul_260, mul_261, sub_78
#   x_303 => mul_262, sigmoid_51
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_417), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_419), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_260, %unsqueeze_421), kwargs = {})
#   %add_183 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_261, %unsqueeze_423), kwargs = {})
#   %sigmoid_51 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_183,), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_183, %sigmoid_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4w/c4wjgl36or56bqo5q7dxpgezwo6uy2jnl7yuxzfxud3tfewnxavk.py
# Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_305 => add_185, mul_264, mul_265, sub_79
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_425), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_427), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_264, %unsqueeze_429), kwargs = {})
#   %add_185 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_265, %unsqueeze_431), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g2/cg2tvkcupuf3k62gkpoejj3dietmw76cg4oh3rvwwqpylx6xx3rz.py
# Topologically Sorted Source Nodes: [x_306], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_306 => convolution_58
# Graph fragment:
#   %convolution_58 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_185, %arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xu/cxuq3yu2p6lard5qefanpindyhm2fwys5szf5yi3sfxind7j3a6d.py
# Topologically Sorted Source Nodes: [x_307, x_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_307 => add_187, mul_267, mul_268, sub_80
#   x_308 => mul_269, sigmoid_52
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_433), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_435), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %unsqueeze_437), kwargs = {})
#   %add_187 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_268, %unsqueeze_439), kwargs = {})
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_187,), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_187, %sigmoid_52), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/li/climknhmrhixj2umxkklokxb72va22xls3w7w6igonyb4qvgn2l3.py
# Topologically Sorted Source Nodes: [layer_norm_26], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_26 => add_188, add_189, mul_270, mul_271, rsqrt_26, sub_81, var_mean_26
# Graph fragment:
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_137, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_137, %getitem_130), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_129, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_188,), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %rsqrt_26), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_270, %arg144_1), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_271, %arg145_1), kwargs = {})
triton_per_fused_native_layer_norm_29 = async_compile.triton('triton_per_fused_native_layer_norm_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_29(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (192*((x0 % 4) % 2)) + (384*((((4*x1) + (x0 % 4)) // 4) % 8)) + (3072*(((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + ((x0 % 4) % 2)) // 16)) + (3072*((x0 % 4) // 2)) + (6144*(((4*x1) + (x0 % 4)) // 32)) + (49152*(x0 // 4)) + (49152*(triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*(((4*x1) + (x0 % 4)) // 32)) + (256*r2) + ((x0 % 4) % 2),  49152))) + (triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*(((4*x1) + (x0 % 4)) // 32)) + ((x0 % 4) % 2),  256))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 192.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (192*x1) + (12288*x0)), tmp27, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtvahzifxgulrs3nf7fcrckpgwrjo4d5sj6vrep4scelrgqoctk.py
# Topologically Sorted Source Nodes: [x_316, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_27 => add_191, add_192, mul_272, mul_273, rsqrt_27, sub_82, var_mean_27
#   x_316 => add_190
# Graph fragment:
#   %add_190 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_137, %view_143), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_190, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_190, %getitem_139), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_138, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_191,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %rsqrt_27), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %arg150_1), kwargs = {})
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %arg151_1), kwargs = {})
triton_per_fused_add_native_layer_norm_30 = async_compile.triton('triton_per_fused_add_native_layer_norm_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (192*((x1 % 4) % 2)) + (384*((((4*x0) + (x1 % 4)) // 4) % 8)) + (3072*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) // 16)) + (3072*((x1 % 4) // 2)) + (6144*(((4*x0) + (x1 % 4)) // 32)) + (49152*(x1 // 4)) + (49152*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*(((4*x0) + (x1 % 4)) // 32)) + (256*r2) + ((x1 % 4) % 2),  49152))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*(((4*x0) + (x1 % 4)) // 32)) + ((x1 % 4) % 2),  256))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (192*x3)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lw/clw4o7zxywkdzxpvkezjx4wpnb2ujdttvm4dtge2a367tahyb5i3.py
# Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   x_318 => mul_274, sigmoid_53
# Graph fragment:
#   %sigmoid_53 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_145,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_145, %sigmoid_53), kwargs = {})
triton_poi_fused_silu_31 = async_compile.triton('triton_poi_fused_silu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_31(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/57/c57tfump2kyvlqqf4k6msw2xpspuutjw52pls2gzclsy4apq4zwz.py
# Topologically Sorted Source Nodes: [x_316, x_322, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_28 => add_194, add_195, mul_275, mul_276, rsqrt_28, sub_83, var_mean_28
#   x_316 => add_190
#   x_322 => add_193
# Graph fragment:
#   %add_190 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_137, %view_143), kwargs = {})
#   %add_193 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %view_147), kwargs = {})
#   %var_mean_28 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_193, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_193, %getitem_141), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_140, 1e-05), kwargs = {})
#   %rsqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_194,), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %rsqrt_28), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %arg156_1), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %arg157_1), kwargs = {})
triton_per_fused_add_native_layer_norm_32 = async_compile.triton('triton_per_fused_add_native_layer_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (192*((x1 % 4) % 2)) + (384*((((4*x0) + (x1 % 4)) // 4) % 8)) + (3072*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) // 16)) + (3072*((x1 % 4) // 2)) + (6144*(((4*x0) + (x1 % 4)) // 32)) + (49152*(x1 // 4)) + (49152*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*(((4*x0) + (x1 % 4)) // 32)) + (256*r2) + ((x1 % 4) % 2),  49152))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*(((4*x0) + (x1 % 4)) // 32)) + ((x1 % 4) % 2),  256))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (192*x3)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/su/csu3eiseffcar6r4fxd7ju3uwakteowcmbstpl4ljcvhepky3eml.py
# Topologically Sorted Source Nodes: [x_327, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_29 => add_197, add_198, mul_277, mul_278, rsqrt_29, sub_84, var_mean_29
#   x_327 => add_196
# Graph fragment:
#   %add_196 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, %view_153), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_196, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_196, %getitem_150), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_149, 1e-05), kwargs = {})
#   %rsqrt_29 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_197,), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %rsqrt_29), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_277, %arg162_1), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_278, %arg163_1), kwargs = {})
triton_per_fused_add_native_layer_norm_33 = async_compile.triton('triton_per_fused_add_native_layer_norm_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5y/c5ybjm5mhhzgankt5u52nmr3pg3rqjnsk4rv4zlfjt3nz6hp7l5x.py
# Topologically Sorted Source Nodes: [x_327, x_333, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_30 => add_200, add_201, mul_280, mul_281, rsqrt_30, sub_85, var_mean_30
#   x_327 => add_196
#   x_333 => add_199
# Graph fragment:
#   %add_196 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_193, %view_153), kwargs = {})
#   %add_199 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_196, %view_157), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_199, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_199, %getitem_152), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_151, 1e-05), kwargs = {})
#   %rsqrt_30 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_200,), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %rsqrt_30), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %arg168_1), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %arg169_1), kwargs = {})
triton_per_fused_add_native_layer_norm_34 = async_compile.triton('triton_per_fused_add_native_layer_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y2/cy2m6l45b4mlf3pxzweacy7hippp27ia4fqyw7qpfdkrdc6cgngu.py
# Topologically Sorted Source Nodes: [x_349, x_355, x_356], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_349 => add_208
#   x_355 => add_211
#   x_356 => var_mean_34
# Graph fragment:
#   %add_208 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_205, %view_173), kwargs = {})
#   %add_211 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_208, %view_177), kwargs = {})
#   %var_mean_34 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_211, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_35 = async_compile.triton('triton_per_fused_add_native_layer_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccyaf5g6bipwaoowhzncibkhovmqrfigegn5cxp4jsvmqlum7dox.py
# Topologically Sorted Source Nodes: [x_359], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_359 => clone_65
# Graph fragment:
#   %clone_65 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_110,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_36 = async_compile.triton('triton_poi_fused_clone_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4, 131072], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 98304
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + ((192*(((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)) + (12288*y0) + (24576*y1) + (49152*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((64*y0) + (128*y1) + (256*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((64*y0) + (128*y1) + (256*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (2*x2) + (16*y1) + (32*x3)), tmp13, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d5/cd5zjpzabicqz25rlwv43x2lu6af7txnflrxtscofxhl66e2nqgn.py
# Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_360 => convolution_60
# Graph fragment:
#   %convolution_60 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_180, %arg194_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_37 = async_compile.triton('triton_poi_fused_convolution_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (16*(x3 % 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y4)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x5) + (49152*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rz/crzlppxm5v5c64qkztbjgfm55qki7hlg2ssbptcvbu3uhohbp656.py
# Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_185, %mul_295], 1), kwargs = {})
triton_poi_fused_cat_38 = async_compile.triton('triton_poi_fused_cat_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_38(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((128*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((128*x1) + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jt/cjtj3htnvka6xy23sgn7oedvtswhrrb7yoroupgtkvt73hyrtpqp.py
# Topologically Sorted Source Nodes: [cat_4, x_363], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   cat_4 => cat_4
#   x_363 => convolution_61
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_185, %mul_295], 1), kwargs = {})
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %arg199_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_39 = async_compile.triton('triton_poi_fused_cat_convolution_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oz/cozzbuoqblu2jasqaqk2whe326uf6ns33izfrzk2ddne6ik3zfqc.py
# Topologically Sorted Source Nodes: [x_367, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_367 => add_219, mul_301, mul_302, sub_92
#   x_368 => mul_303, sigmoid_59
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_457), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_459), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_461), kwargs = {})
#   %add_219 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %unsqueeze_463), kwargs = {})
#   %sigmoid_59 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_219,), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_219, %sigmoid_59), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/it/citxqlvv5dp3wnfqi275k7dwxp3cp4tput7ergonknx7wussvwin.py
# Topologically Sorted Source Nodes: [x_370, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_370 => add_221, mul_305, mul_306, sub_93
#   x_371 => mul_307, sigmoid_60
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_465), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_467), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %unsqueeze_469), kwargs = {})
#   %add_221 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %unsqueeze_471), kwargs = {})
#   %sigmoid_60 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_221,), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_221, %sigmoid_60), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/72/c72rdlqb5fhxiefohuzhegurhvccffulkykrvevp6sbv2g4ig3q2.py
# Topologically Sorted Source Nodes: [x_373], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_373 => add_223, mul_309, mul_310, sub_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_473), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_475), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, %unsqueeze_477), kwargs = {})
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_310, %unsqueeze_479), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gl/cglrjd3sppguggfpvvfz7nlymbfotkdajc7cfux2ffstsgerwf6j.py
# Topologically Sorted Source Nodes: [x_374], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_374 => convolution_65
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_223, %arg219_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_43 = async_compile.triton('triton_poi_fused_convolution_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_43(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1440*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rf/crfnh2scbu5exma66wmdge6q67esbq3cz3sqbz4y7ebqsfahrceu.py
# Topologically Sorted Source Nodes: [x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
# Source node to ATen node mapping:
#   x_375 => add_225, mul_312, mul_313, sub_95
#   x_376 => mul_314, sigmoid_61
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_65, %unsqueeze_481), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_483), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_312, %unsqueeze_485), kwargs = {})
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_313, %unsqueeze_487), kwargs = {})
#   %sigmoid_61 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_225,), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_225, %sigmoid_61), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_silu_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_silu_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_silu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_silu_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2g/c2giic7uds4grr6gfikrtmskc77lfv2htgw5nnzuywibtgkecfdc.py
# Topologically Sorted Source Nodes: [layer_norm_35], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_35 => add_226, add_227, mul_315, mul_316, rsqrt_35, sub_96, var_mean_35
# Graph fragment:
#   %var_mean_35 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_183, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_183, %getitem_176), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_175, 1e-05), kwargs = {})
#   %rsqrt_35 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_226,), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %rsqrt_35), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_315, %arg225_1), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_316, %arg226_1), kwargs = {})
triton_per_fused_native_layer_norm_45 = async_compile.triton('triton_per_fused_native_layer_norm_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_45(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (240*((x0 % 4) % 2)) + (480*((((4*x1) + (x0 % 4)) // 4) % 4)) + (1920*(((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + ((x0 % 4) % 2)) // 8)) + (1920*((x0 % 4) // 2)) + (3840*(((4*x1) + (x0 % 4)) // 16)) + (15360*(x0 // 4)) + (15360*(triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*(((4*x1) + (x0 % 4)) // 16)) + (64*r2) + ((x0 % 4) % 2),  15360))) + (triton_helpers.div_floor_integer((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*(((4*x1) + (x0 % 4)) // 16)) + ((x0 % 4) % 2),  64))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 240.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r2 + (240*x1) + (3840*x0)), tmp27, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/is/ciswdak4zpyglrajsozj4olfk6epruopge2v6o3p6s6iree3r7ds.py
# Topologically Sorted Source Nodes: [x_384, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_36 => add_229, add_230, mul_317, mul_318, rsqrt_36, sub_97, var_mean_36
#   x_384 => add_228
# Graph fragment:
#   %add_228 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_183, %view_189), kwargs = {})
#   %var_mean_36 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_228, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_228, %getitem_185), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_184, 1e-05), kwargs = {})
#   %rsqrt_36 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_229,), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %rsqrt_36), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, %arg231_1), kwargs = {})
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %arg232_1), kwargs = {})
triton_per_fused_add_native_layer_norm_46 = async_compile.triton('triton_per_fused_add_native_layer_norm_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (240*((x1 % 4) % 2)) + (480*((((4*x0) + (x1 % 4)) // 4) % 4)) + (1920*(((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + ((x1 % 4) % 2)) // 8)) + (1920*((x1 % 4) // 2)) + (3840*(((4*x0) + (x1 % 4)) // 16)) + (15360*(x1 // 4)) + (15360*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*(((4*x0) + (x1 % 4)) // 16)) + (64*r2) + ((x1 % 4) % 2),  15360))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*(((4*x0) + (x1 % 4)) // 16)) + ((x1 % 4) % 2),  64))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (240*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 240.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r2 + (240*x3)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aj/cajze5p466dslk2moowxrdwhqnkgjcpg36s2csbm7jnqsqeg3noq.py
# Topologically Sorted Source Nodes: [x_386], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   x_386 => mul_319, sigmoid_62
# Graph fragment:
#   %sigmoid_62 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_191,), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_191, %sigmoid_62), kwargs = {})
triton_poi_fused_silu_47 = async_compile.triton('triton_poi_fused_silu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_47(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sa/csawtjpawczr25kbb23n4cwl3jvf36mjpqunml76qsvjfzfl3k2b.py
# Topologically Sorted Source Nodes: [x_384, x_390, layer_norm_37], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_37 => add_232, add_233, mul_320, mul_321, rsqrt_37, sub_98, var_mean_37
#   x_384 => add_228
#   x_390 => add_231
# Graph fragment:
#   %add_228 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_183, %view_189), kwargs = {})
#   %add_231 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_228, %view_193), kwargs = {})
#   %var_mean_37 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_231, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_231, %getitem_187), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_186, 1e-05), kwargs = {})
#   %rsqrt_37 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_232,), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %rsqrt_37), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_320, %arg237_1), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_321, %arg238_1), kwargs = {})
triton_per_fused_add_native_layer_norm_48 = async_compile.triton('triton_per_fused_add_native_layer_norm_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (240*((x1 % 4) % 2)) + (480*((((4*x0) + (x1 % 4)) // 4) % 4)) + (1920*(((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + ((x1 % 4) % 2)) // 8)) + (1920*((x1 % 4) // 2)) + (3840*(((4*x0) + (x1 % 4)) // 16)) + (15360*(x1 // 4)) + (15360*(triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*(((4*x0) + (x1 % 4)) // 16)) + (64*r2) + ((x1 % 4) % 2),  15360))) + (triton_helpers.div_floor_integer((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*(((4*x0) + (x1 % 4)) // 16)) + ((x1 % 4) % 2),  64))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (240*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + (240*x3)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 240.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r2 + (240*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (240*x3)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j6/cj62i7knazi7dh2i4d47hyvbslso3z2w5mx4yp4molsdxh5kzdwy.py
# Topologically Sorted Source Nodes: [x_395, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_38 => add_235, add_236, mul_322, mul_323, rsqrt_38, sub_99, var_mean_38
#   x_395 => add_234
# Graph fragment:
#   %add_234 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_231, %view_199), kwargs = {})
#   %var_mean_38 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_234, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_234, %getitem_196), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_195, 1e-05), kwargs = {})
#   %rsqrt_38 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_235,), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %rsqrt_38), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %arg243_1), kwargs = {})
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %arg244_1), kwargs = {})
triton_per_fused_add_native_layer_norm_49 = async_compile.triton('triton_per_fused_add_native_layer_norm_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 240.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/go/cgobxbatodssfow6pci7ezse2h5a2fx2ktt2xy75c7lspsm2h26s.py
# Topologically Sorted Source Nodes: [x_395, x_401, layer_norm_39], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_39 => add_238, add_239, mul_325, mul_326, rsqrt_39, sub_100, var_mean_39
#   x_395 => add_234
#   x_401 => add_237
# Graph fragment:
#   %add_234 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_231, %view_199), kwargs = {})
#   %add_237 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_234, %view_203), kwargs = {})
#   %var_mean_39 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_237, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_237, %getitem_198), kwargs = {})
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_197, 1e-05), kwargs = {})
#   %rsqrt_39 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_238,), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %rsqrt_39), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %arg249_1), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %arg250_1), kwargs = {})
triton_per_fused_add_native_layer_norm_50 = async_compile.triton('triton_per_fused_add_native_layer_norm_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 240.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4z/c4z5zje4qqw4vyv7ntj4dixp42fhwdh7y2vegdintnkpi6snviwd.py
# Topologically Sorted Source Nodes: [x_406, x_412, x_413], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_406 => add_240
#   x_412 => add_243
#   x_413 => var_mean_41
# Graph fragment:
#   %add_240 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_237, %view_209), kwargs = {})
#   %add_243 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_240, %view_213), kwargs = {})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_243, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_51 = async_compile.triton('triton_per_fused_add_native_layer_norm_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/f4/cf4cjydn4bvm7ynjyxzblwtu77iy2ih7uuisk64o7k7vmbggildp.py
# Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_416 => clone_78
# Graph fragment:
#   %clone_78 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_132,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_52 = async_compile.triton('triton_poi_fused_clone_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 30720
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 4
    x3 = (xindex // 4)
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + ((240*(((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)) + (3840*y0) + (7680*y1) + (15360*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((16*y0) + (32*y1) + (64*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((16*y0) + (32*y1) + (64*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 240.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (y0 + (2*x2) + (8*y1) + (16*x3)), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pz/cpzzcwaixprsydjmdwkck6sx2eytx3qhvidlee7bhmeedyug6izp.py
# Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_417 => convolution_67
# Graph fragment:
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_216, %arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_53 = async_compile.triton('triton_poi_fused_convolution_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_ptr0 + (x2 + (8*(x3 % 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y4)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (240*x5) + (15360*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vb/cvbvjeglhhr5mmpx2mtrovdqqswejik5qznykdisbxtntxra6acc.py
# Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_5 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_223, %mul_335], 1), kwargs = {})
triton_poi_fused_cat_54 = async_compile.triton('triton_poi_fused_cat_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 320
    x1 = (xindex // 320)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((160*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 320, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((160*x1) + ((-160) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e7/ce7qysadq7fywrlpcbvdfivl3dgkbhzaqjqae4geeuenbnpyixpo.py
# Topologically Sorted Source Nodes: [cat_5, x_420], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   cat_5 => cat_5
#   x_420 => convolution_68
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_223, %mul_335], 1), kwargs = {})
#   %convolution_68 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_5, %arg268_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_55 = async_compile.triton('triton_poi_fused_cat_convolution_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 51200
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (320*x2) + (2880*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpqizkwqztop7rxzyytnveke7yhph3jjy4rjkmehpp5w24naaarm.py
# Topologically Sorted Source Nodes: [x_424], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_424 => add_251, mul_341, mul_342, sub_105
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_505), kwargs = {})
#   %mul_341 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_507), kwargs = {})
#   %mul_342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_341, %unsqueeze_509), kwargs = {})
#   %add_251 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_342, %unsqueeze_511), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/47/c47sfjiy5tblhimhdj2u4gqehq3el6fyocbuw6wjwzzyydgtdc7b.py
# Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.silu, aten.mean]
# Source node to ATen node mapping:
#   x_425 => mul_343, sigmoid_67
#   x_426 => mean_1
# Graph fragment:
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_251,), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_251, %sigmoid_67), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_343, [-1, -2], True), kwargs = {})
triton_per_fused_mean_silu_57 = async_compile.triton('triton_per_fused_mean_silu_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_silu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_silu_57(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 640
    x1 = (xindex // 640)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r2) + (40960*x1)), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (32, ), (1, ))
    assert_size_stride(arg21_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, ), (1, ))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, ), (1, ))
    assert_size_stride(arg76_1, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg77_1, (96, ), (1, ))
    assert_size_stride(arg78_1, (96, ), (1, ))
    assert_size_stride(arg79_1, (96, ), (1, ))
    assert_size_stride(arg80_1, (96, ), (1, ))
    assert_size_stride(arg81_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg82_1, (96, ), (1, ))
    assert_size_stride(arg83_1, (96, ), (1, ))
    assert_size_stride(arg84_1, (96, ), (1, ))
    assert_size_stride(arg85_1, (96, ), (1, ))
    assert_size_stride(arg86_1, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg87_1, (144, ), (1, ))
    assert_size_stride(arg88_1, (144, ), (1, ))
    assert_size_stride(arg89_1, (432, 144), (144, 1))
    assert_size_stride(arg90_1, (432, ), (1, ))
    assert_size_stride(arg91_1, (144, 144), (144, 1))
    assert_size_stride(arg92_1, (144, ), (1, ))
    assert_size_stride(arg93_1, (144, ), (1, ))
    assert_size_stride(arg94_1, (144, ), (1, ))
    assert_size_stride(arg95_1, (288, 144), (144, 1))
    assert_size_stride(arg96_1, (288, ), (1, ))
    assert_size_stride(arg97_1, (144, 288), (288, 1))
    assert_size_stride(arg98_1, (144, ), (1, ))
    assert_size_stride(arg99_1, (144, ), (1, ))
    assert_size_stride(arg100_1, (144, ), (1, ))
    assert_size_stride(arg101_1, (432, 144), (144, 1))
    assert_size_stride(arg102_1, (432, ), (1, ))
    assert_size_stride(arg103_1, (144, 144), (144, 1))
    assert_size_stride(arg104_1, (144, ), (1, ))
    assert_size_stride(arg105_1, (144, ), (1, ))
    assert_size_stride(arg106_1, (144, ), (1, ))
    assert_size_stride(arg107_1, (288, 144), (144, 1))
    assert_size_stride(arg108_1, (288, ), (1, ))
    assert_size_stride(arg109_1, (144, 288), (288, 1))
    assert_size_stride(arg110_1, (144, ), (1, ))
    assert_size_stride(arg111_1, (144, ), (1, ))
    assert_size_stride(arg112_1, (144, ), (1, ))
    assert_size_stride(arg113_1, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg114_1, (96, ), (1, ))
    assert_size_stride(arg115_1, (96, ), (1, ))
    assert_size_stride(arg116_1, (96, ), (1, ))
    assert_size_stride(arg117_1, (96, ), (1, ))
    assert_size_stride(arg118_1, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg119_1, (96, ), (1, ))
    assert_size_stride(arg120_1, (96, ), (1, ))
    assert_size_stride(arg121_1, (96, ), (1, ))
    assert_size_stride(arg122_1, (96, ), (1, ))
    assert_size_stride(arg123_1, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, ), (1, ))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (576, 192), (192, 1))
    assert_size_stride(arg147_1, (576, ), (1, ))
    assert_size_stride(arg148_1, (192, 192), (192, 1))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (384, 192), (192, 1))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (192, 384), (384, 1))
    assert_size_stride(arg155_1, (192, ), (1, ))
    assert_size_stride(arg156_1, (192, ), (1, ))
    assert_size_stride(arg157_1, (192, ), (1, ))
    assert_size_stride(arg158_1, (576, 192), (192, 1))
    assert_size_stride(arg159_1, (576, ), (1, ))
    assert_size_stride(arg160_1, (192, 192), (192, 1))
    assert_size_stride(arg161_1, (192, ), (1, ))
    assert_size_stride(arg162_1, (192, ), (1, ))
    assert_size_stride(arg163_1, (192, ), (1, ))
    assert_size_stride(arg164_1, (384, 192), (192, 1))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (192, 384), (384, 1))
    assert_size_stride(arg167_1, (192, ), (1, ))
    assert_size_stride(arg168_1, (192, ), (1, ))
    assert_size_stride(arg169_1, (192, ), (1, ))
    assert_size_stride(arg170_1, (576, 192), (192, 1))
    assert_size_stride(arg171_1, (576, ), (1, ))
    assert_size_stride(arg172_1, (192, 192), (192, 1))
    assert_size_stride(arg173_1, (192, ), (1, ))
    assert_size_stride(arg174_1, (192, ), (1, ))
    assert_size_stride(arg175_1, (192, ), (1, ))
    assert_size_stride(arg176_1, (384, 192), (192, 1))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (192, 384), (384, 1))
    assert_size_stride(arg179_1, (192, ), (1, ))
    assert_size_stride(arg180_1, (192, ), (1, ))
    assert_size_stride(arg181_1, (192, ), (1, ))
    assert_size_stride(arg182_1, (576, 192), (192, 1))
    assert_size_stride(arg183_1, (576, ), (1, ))
    assert_size_stride(arg184_1, (192, 192), (192, 1))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (384, 192), (192, 1))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (192, 384), (384, 1))
    assert_size_stride(arg191_1, (192, ), (1, ))
    assert_size_stride(arg192_1, (192, ), (1, ))
    assert_size_stride(arg193_1, (192, ), (1, ))
    assert_size_stride(arg194_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg215_1, (160, ), (1, ))
    assert_size_stride(arg216_1, (160, ), (1, ))
    assert_size_stride(arg217_1, (160, ), (1, ))
    assert_size_stride(arg218_1, (160, ), (1, ))
    assert_size_stride(arg219_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg220_1, (160, ), (1, ))
    assert_size_stride(arg221_1, (160, ), (1, ))
    assert_size_stride(arg222_1, (160, ), (1, ))
    assert_size_stride(arg223_1, (160, ), (1, ))
    assert_size_stride(arg224_1, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg225_1, (240, ), (1, ))
    assert_size_stride(arg226_1, (240, ), (1, ))
    assert_size_stride(arg227_1, (720, 240), (240, 1))
    assert_size_stride(arg228_1, (720, ), (1, ))
    assert_size_stride(arg229_1, (240, 240), (240, 1))
    assert_size_stride(arg230_1, (240, ), (1, ))
    assert_size_stride(arg231_1, (240, ), (1, ))
    assert_size_stride(arg232_1, (240, ), (1, ))
    assert_size_stride(arg233_1, (480, 240), (240, 1))
    assert_size_stride(arg234_1, (480, ), (1, ))
    assert_size_stride(arg235_1, (240, 480), (480, 1))
    assert_size_stride(arg236_1, (240, ), (1, ))
    assert_size_stride(arg237_1, (240, ), (1, ))
    assert_size_stride(arg238_1, (240, ), (1, ))
    assert_size_stride(arg239_1, (720, 240), (240, 1))
    assert_size_stride(arg240_1, (720, ), (1, ))
    assert_size_stride(arg241_1, (240, 240), (240, 1))
    assert_size_stride(arg242_1, (240, ), (1, ))
    assert_size_stride(arg243_1, (240, ), (1, ))
    assert_size_stride(arg244_1, (240, ), (1, ))
    assert_size_stride(arg245_1, (480, 240), (240, 1))
    assert_size_stride(arg246_1, (480, ), (1, ))
    assert_size_stride(arg247_1, (240, 480), (480, 1))
    assert_size_stride(arg248_1, (240, ), (1, ))
    assert_size_stride(arg249_1, (240, ), (1, ))
    assert_size_stride(arg250_1, (240, ), (1, ))
    assert_size_stride(arg251_1, (720, 240), (240, 1))
    assert_size_stride(arg252_1, (720, ), (1, ))
    assert_size_stride(arg253_1, (240, 240), (240, 1))
    assert_size_stride(arg254_1, (240, ), (1, ))
    assert_size_stride(arg255_1, (240, ), (1, ))
    assert_size_stride(arg256_1, (240, ), (1, ))
    assert_size_stride(arg257_1, (480, 240), (240, 1))
    assert_size_stride(arg258_1, (480, ), (1, ))
    assert_size_stride(arg259_1, (240, 480), (480, 1))
    assert_size_stride(arg260_1, (240, ), (1, ))
    assert_size_stride(arg261_1, (240, ), (1, ))
    assert_size_stride(arg262_1, (240, ), (1, ))
    assert_size_stride(arg263_1, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg264_1, (160, ), (1, ))
    assert_size_stride(arg265_1, (160, ), (1, ))
    assert_size_stride(arg266_1, (160, ), (1, ))
    assert_size_stride(arg267_1, (160, ), (1, ))
    assert_size_stride(arg268_1, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(arg269_1, (160, ), (1, ))
    assert_size_stride(arg270_1, (160, ), (1, ))
    assert_size_stride(arg271_1, (160, ), (1, ))
    assert_size_stride(arg272_1, (160, ), (1, ))
    assert_size_stride(arg273_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg274_1, (640, ), (1, ))
    assert_size_stride(arg275_1, (640, ), (1, ))
    assert_size_stride(arg276_1, (640, ), (1, ))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (1000, 640), (640, 1))
    assert_size_stride(arg279_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 256, 256), (196608, 1, 768, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 65536, grid=grid(24, 65536), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 128, 128), (262144, 1, 2048, 16))
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((8, 16, 128, 128), (262144, 1, 2048, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_216, x_217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 2097152, grid=grid(2097152), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf3
        # Topologically Sorted Source Nodes: [x_217, x_218], Original ATen: [aten.silu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del arg6_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        buf7 = empty_strided_cuda((8, 64, 128, 128), (1048576, 1, 8192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_3.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, buf7, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf6
        # Topologically Sorted Source Nodes: [x_220, x_221], Original ATen: [aten.silu, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf8, (8, 64, 128, 128), (1048576, 1, 8192, 64))
        del arg11_1
        buf9 = buf8; del buf8  # reuse
        buf10 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_222, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_3.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, buf10, 8388608, grid=grid(8388608), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf9
        # Topologically Sorted Source Nodes: [x_223, x_224], Original ATen: [aten.silu, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 32, 128, 128), (524288, 1, 4096, 32))
        del arg16_1
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [x_225, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 128, 128, 128), (2097152, 1, 16384, 128))
        del arg21_1
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((8, 128, 128, 128), (2097152, 1, 16384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_227, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_5.run(buf14, arg22_1, arg23_1, arg24_1, arg25_1, buf15, 16777216, grid=grid(16777216), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf14
        # Topologically Sorted Source Nodes: [x_228, x_229], Original ATen: [aten.silu, aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg26_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf16, (8, 128, 64, 64), (524288, 1, 8192, 128))
        del arg26_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        buf18 = reinterpret_tensor(buf12, (8, 128, 64, 64), (524288, 1, 8192, 128), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_6.run(buf17, arg27_1, arg28_1, arg29_1, arg30_1, buf18, 4194304, grid=grid(4194304), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf17
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten.silu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg31_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf20, arg32_1, arg33_1, arg34_1, arg35_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg36_1
        buf22 = buf21; del buf21  # reuse
        buf23 = reinterpret_tensor(buf10, (8, 256, 64, 64), (1048576, 1, 16384, 256), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf22, arg37_1, arg38_1, arg39_1, arg40_1, buf23, 8388608, grid=grid(8388608), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf22
        # Topologically Sorted Source Nodes: [x_236, x_237], Original ATen: [aten.silu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg41_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf24, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg41_1
        buf25 = buf24; del buf24  # reuse
        buf26 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf25, arg42_1, arg43_1, arg44_1, arg45_1, buf26, 8388608, grid=grid(8388608), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf25
        # Topologically Sorted Source Nodes: [x_239, x_240], Original ATen: [aten.silu, aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg46_1
        buf28 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf28, buf27, arg47_1, arg48_1, arg49_1, arg50_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del buf27
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg51_1
        buf30 = buf29; del buf29  # reuse
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_244, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf30, arg52_1, arg53_1, arg54_1, arg55_1, buf31, 8388608, grid=grid(8388608), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf30
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten.silu, aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg56_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf32, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg56_1
        buf33 = buf32; del buf32  # reuse
        buf34 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf33, arg57_1, arg58_1, arg59_1, arg60_1, buf34, 8388608, grid=grid(8388608), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf33
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten.silu, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 64, 64, 64), (262144, 1, 4096, 64))
        del arg61_1
        buf36 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_250, x_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf36, buf35, arg62_1, arg63_1, arg64_1, arg65_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        del buf35
        # Topologically Sorted Source Nodes: [x_250, x_251, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 256, 64, 64), (1048576, 1, 16384, 256))
        del arg66_1
        buf38 = buf37; del buf37  # reuse
        buf39 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_8.run(buf38, arg67_1, arg68_1, arg69_1, arg70_1, buf39, 8388608, grid=grid(8388608), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf38
        # Topologically Sorted Source Nodes: [x_254, x_255], Original ATen: [aten.silu, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg71_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf40, (8, 256, 32, 32), (262144, 1, 8192, 256))
        del arg71_1
        del buf39
        buf41 = buf40; del buf40  # reuse
        buf42 = reinterpret_tensor(buf36, (8, 256, 32, 32), (262144, 1, 8192, 256), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_10.run(buf41, arg72_1, arg73_1, arg74_1, arg75_1, buf42, 2097152, grid=grid(2097152), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf41
        # Topologically Sorted Source Nodes: [x_257, x_258], Original ATen: [aten.silu, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 96, 32, 32), (98304, 1, 3072, 96))
        del arg76_1
        del buf42
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf44, arg77_1, arg78_1, arg79_1, arg80_1, 786432, grid=grid(786432), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf45 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(arg81_1, buf45, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [x_260], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 96, 32, 32), (98304, 1, 3072, 96))
        del buf45
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided_cuda((8, 96, 32, 32), (98304, 1, 3072, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_261, x_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf47, arg82_1, arg83_1, arg84_1, arg85_1, buf48, 786432, grid=grid(786432), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf47
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten.silu, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 144, 32, 32), (147456, 1, 4608, 144))
        del arg86_1
        del buf48
        buf53 = empty_strided_cuda((32, 256, 144), (36864, 144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_21], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_14.run(buf49, arg87_1, arg88_1, buf53, 8192, 144, grid=grid(8192), stream=stream0)
        del arg87_1
        del arg88_1
        buf54 = empty_strided_cuda((8192, 432), (432, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf53, (8192, 144), (144, 1), 0), reinterpret_tensor(arg89_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf54)
        del arg89_1
        del arg90_1
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf55 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf54, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, False)
        buf56 = buf55[0]
        del buf55
        buf60 = reinterpret_tensor(buf53, (8192, 144), (144, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf56, (8192, 144), (144, 1), 0), reinterpret_tensor(arg91_1, (144, 144), (1, 144), 0), out=buf60)
        del arg91_1
        buf64 = reinterpret_tensor(buf56, (32, 256, 144), (36864, 144, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_270, layer_norm_22], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_15.run(buf49, buf60, arg92_1, arg93_1, arg94_1, buf64, 8192, 144, grid=grid(8192), stream=stream0)
        del arg93_1
        del arg94_1
        buf65 = empty_strided_cuda((8192, 288), (288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf64, (8192, 144), (144, 1), 0), reinterpret_tensor(arg95_1, (144, 288), (1, 144), 0), out=buf65)
        del arg95_1
        buf66 = reinterpret_tensor(buf65, (32, 256, 288), (73728, 288, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten.silu]
        triton_poi_fused_silu_16.run(buf66, arg96_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg96_1
        buf67 = reinterpret_tensor(buf64, (8192, 144), (144, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf66, (8192, 288), (288, 1), 0), reinterpret_tensor(arg97_1, (288, 144), (1, 288), 0), out=buf67)
        del arg97_1
        buf68 = reinterpret_tensor(buf67, (32, 256, 144), (36864, 144, 1), 0); del buf67  # reuse
        buf72 = empty_strided_cuda((32, 256, 144), (36864, 144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_270, x_276, layer_norm_23], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_17.run(buf68, buf49, buf60, arg92_1, arg98_1, arg99_1, arg100_1, buf72, 8192, 144, grid=grid(8192), stream=stream0)
        del arg100_1
        del arg92_1
        del arg98_1
        del arg99_1
        del buf49
        del buf60
        buf73 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg102_1, reinterpret_tensor(buf72, (8192, 144), (144, 1), 0), reinterpret_tensor(arg101_1, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf73)
        del arg101_1
        del arg102_1
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf74 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf73, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf73, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf73, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, False)
        del buf73
        buf75 = buf74[0]
        del buf74
        buf79 = reinterpret_tensor(buf72, (8192, 144), (144, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf75, (8192, 144), (144, 1), 0), reinterpret_tensor(arg103_1, (144, 144), (1, 144), 0), out=buf79)
        del arg103_1
        buf83 = reinterpret_tensor(buf75, (32, 256, 144), (36864, 144, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_281, layer_norm_24], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf68, buf79, arg104_1, arg105_1, arg106_1, buf83, 8192, 144, grid=grid(8192), stream=stream0)
        del arg105_1
        del arg106_1
        buf84 = reinterpret_tensor(buf66, (8192, 288), (288, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf83, (8192, 144), (144, 1), 0), reinterpret_tensor(arg107_1, (144, 288), (1, 144), 0), out=buf84)
        del arg107_1
        buf85 = reinterpret_tensor(buf84, (32, 256, 288), (73728, 288, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten.silu]
        triton_poi_fused_silu_16.run(buf85, arg108_1, 2359296, grid=grid(2359296), stream=stream0)
        del arg108_1
        buf86 = reinterpret_tensor(buf83, (8192, 144), (144, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf85, (8192, 288), (288, 1), 0), reinterpret_tensor(arg109_1, (288, 144), (1, 288), 0), out=buf86)
        del arg109_1
        del buf85
        buf87 = reinterpret_tensor(buf86, (32, 256, 144), (36864, 144, 1), 0); del buf86  # reuse
        buf88 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf89 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [x_281, x_287, x_288], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf87, buf68, buf79, arg104_1, arg110_1, buf88, buf89, 8192, 144, grid=grid(8192), stream=stream0)
        del arg104_1
        del arg110_1
        del buf68
        buf91 = reinterpret_tensor(buf79, (18432, 2, 16, 2), (64, 32, 2, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf87, buf88, buf89, arg111_1, arg112_1, buf91, 4, 294912, grid=grid(4, 294912), stream=stream0)
        del arg111_1
        del arg112_1
        del buf88
        del buf89
        buf92 = reinterpret_tensor(buf87, (8, 144, 32, 32), (147456, 1, 4608, 144), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf91, buf92, 1152, 1024, grid=grid(1152, 1024), stream=stream0)
        del buf91
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 96, 32, 32), (98304, 1, 3072, 96))
        del arg113_1
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_293], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf94, arg114_1, arg115_1, arg116_1, arg117_1, 786432, grid=grid(786432), stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf95 = reinterpret_tensor(buf0, (8, 192, 32, 32), (196608, 1, 6144, 192), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [cat_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_22.run(buf44, buf94, buf95, 1572864, grid=grid(1572864), stream=stream0)
        del buf44
        buf96 = empty_strided_cuda((96, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [cat_3, x_295], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_23.run(arg118_1, buf96, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg118_1
        # Topologically Sorted Source Nodes: [cat_3, x_295], Original ATen: [aten.cat, aten.convolution]
        buf97 = extern_kernels.convolution(buf95, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 96, 32, 32), (98304, 1, 3072, 96))
        del buf95
        del buf96
        buf98 = buf97; del buf97  # reuse
        buf99 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_13.run(buf98, arg119_1, arg120_1, arg121_1, arg122_1, buf99, 786432, grid=grid(786432), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        del arg122_1
        del buf98
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten.silu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 384, 32, 32), (393216, 1, 12288, 384))
        del arg123_1
        buf101 = buf100; del buf100  # reuse
        buf102 = empty_strided_cuda((8, 384, 32, 32), (393216, 1, 12288, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_299, x_300], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_24.run(buf101, arg124_1, arg125_1, arg126_1, arg127_1, buf102, 3145728, grid=grid(3145728), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        del buf101
        # Topologically Sorted Source Nodes: [x_300, x_301], Original ATen: [aten.silu, aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg128_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf103, (8, 384, 16, 16), (98304, 1, 6144, 384))
        del arg128_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        buf105 = reinterpret_tensor(buf99, (8, 384, 16, 16), (98304, 1, 6144, 384), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_25.run(buf104, arg129_1, arg130_1, arg131_1, arg132_1, buf105, 786432, grid=grid(786432), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        del buf104
        # Topologically Sorted Source Nodes: [x_303, x_304], Original ATen: [aten.silu, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 128, 16, 16), (32768, 1, 2048, 128))
        del arg133_1
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_26.run(buf107, arg134_1, arg135_1, arg136_1, arg137_1, 262144, grid=grid(262144), stream=stream0)
        del arg134_1
        del arg135_1
        del arg136_1
        del arg137_1
        buf108 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_306], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg138_1, buf108, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg138_1
        # Topologically Sorted Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf107, buf108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 128, 16, 16), (32768, 1, 2048, 128))
        del buf108
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided_cuda((8, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_307, x_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf110, arg139_1, arg140_1, arg141_1, arg142_1, buf111, 262144, grid=grid(262144), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del buf110
        # Topologically Sorted Source Nodes: [x_308, x_309], Original ATen: [aten.silu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 192, 16, 16), (49152, 1, 3072, 192))
        del arg143_1
        del buf111
        buf116 = empty_strided_cuda((32, 64, 192), (12288, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_26], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_29.run(buf112, arg144_1, arg145_1, buf116, 2048, 192, grid=grid(2048), stream=stream0)
        del arg144_1
        del arg145_1
        buf117 = reinterpret_tensor(buf92, (2048, 576), (576, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg147_1, reinterpret_tensor(buf116, (2048, 192), (192, 1), 0), reinterpret_tensor(arg146_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf117)
        del arg146_1
        del arg147_1
        # Topologically Sorted Source Nodes: [x_312], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf118 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf117, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf117, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf117, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf119 = buf118[0]
        del buf118
        buf123 = reinterpret_tensor(buf116, (2048, 192), (192, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf119, (2048, 192), (192, 1), 0), reinterpret_tensor(arg148_1, (192, 192), (1, 192), 0), out=buf123)
        del arg148_1
        buf127 = reinterpret_tensor(buf119, (32, 64, 192), (12288, 192, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_316, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_30.run(buf112, buf123, arg149_1, arg150_1, arg151_1, buf127, 2048, 192, grid=grid(2048), stream=stream0)
        del arg150_1
        del arg151_1
        buf128 = reinterpret_tensor(buf105, (2048, 384), (384, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf127, (2048, 192), (192, 1), 0), reinterpret_tensor(arg152_1, (192, 384), (1, 192), 0), out=buf128)
        del arg152_1
        buf129 = reinterpret_tensor(buf128, (32, 64, 384), (24576, 384, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf129, arg153_1, 786432, grid=grid(786432), stream=stream0)
        del arg153_1
        buf130 = reinterpret_tensor(buf127, (2048, 192), (192, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf129, (2048, 384), (384, 1), 0), reinterpret_tensor(arg154_1, (384, 192), (1, 384), 0), out=buf130)
        del arg154_1
        buf131 = reinterpret_tensor(buf130, (32, 64, 192), (12288, 192, 1), 0); del buf130  # reuse
        buf135 = empty_strided_cuda((32, 64, 192), (12288, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_316, x_322, layer_norm_28], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_32.run(buf131, buf112, buf123, arg149_1, arg155_1, arg156_1, arg157_1, buf135, 2048, 192, grid=grid(2048), stream=stream0)
        del arg149_1
        del arg155_1
        del arg156_1
        del arg157_1
        del buf112
        buf136 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg159_1, reinterpret_tensor(buf135, (2048, 192), (192, 1), 0), reinterpret_tensor(arg158_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf136)
        del arg158_1
        del arg159_1
        # Topologically Sorted Source Nodes: [x_323], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf137 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf136, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf138 = buf137[0]
        del buf137
        buf142 = reinterpret_tensor(buf135, (2048, 192), (192, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf138, (2048, 192), (192, 1), 0), reinterpret_tensor(arg160_1, (192, 192), (1, 192), 0), out=buf142)
        del arg160_1
        buf146 = reinterpret_tensor(buf138, (32, 64, 192), (12288, 192, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [x_327, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf131, buf142, arg161_1, arg162_1, arg163_1, buf146, 2048, 192, grid=grid(2048), stream=stream0)
        del arg162_1
        del arg163_1
        buf147 = reinterpret_tensor(buf129, (2048, 384), (384, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf146, (2048, 192), (192, 1), 0), reinterpret_tensor(arg164_1, (192, 384), (1, 192), 0), out=buf147)
        del arg164_1
        buf148 = reinterpret_tensor(buf147, (32, 64, 384), (24576, 384, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_329], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf148, arg165_1, 786432, grid=grid(786432), stream=stream0)
        del arg165_1
        buf149 = reinterpret_tensor(buf146, (2048, 192), (192, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf148, (2048, 384), (384, 1), 0), reinterpret_tensor(arg166_1, (384, 192), (1, 384), 0), out=buf149)
        del arg166_1
        buf150 = reinterpret_tensor(buf149, (32, 64, 192), (12288, 192, 1), 0); del buf149  # reuse
        buf154 = reinterpret_tensor(buf123, (32, 64, 192), (12288, 192, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_327, x_333, layer_norm_30], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf150, buf131, buf142, arg161_1, arg167_1, arg168_1, arg169_1, buf154, 2048, 192, grid=grid(2048), stream=stream0)
        del arg161_1
        del arg167_1
        del arg168_1
        del arg169_1
        del buf131
        buf155 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg171_1, reinterpret_tensor(buf154, (2048, 192), (192, 1), 0), reinterpret_tensor(arg170_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf155)
        del arg170_1
        del arg171_1
        # Topologically Sorted Source Nodes: [x_334], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf155, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        buf157 = buf156[0]
        del buf156
        buf161 = reinterpret_tensor(buf154, (2048, 192), (192, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (2048, 192), (192, 1), 0), reinterpret_tensor(arg172_1, (192, 192), (1, 192), 0), out=buf161)
        del arg172_1
        buf165 = reinterpret_tensor(buf157, (32, 64, 192), (12288, 192, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_338, layer_norm_31], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf150, buf161, arg173_1, arg174_1, arg175_1, buf165, 2048, 192, grid=grid(2048), stream=stream0)
        del arg174_1
        del arg175_1
        buf166 = reinterpret_tensor(buf148, (2048, 384), (384, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 192), (192, 1), 0), reinterpret_tensor(arg176_1, (192, 384), (1, 192), 0), out=buf166)
        del arg176_1
        buf167 = reinterpret_tensor(buf166, (32, 64, 384), (24576, 384, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_340], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf167, arg177_1, 786432, grid=grid(786432), stream=stream0)
        del arg177_1
        buf168 = reinterpret_tensor(buf165, (2048, 192), (192, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf167, (2048, 384), (384, 1), 0), reinterpret_tensor(arg178_1, (384, 192), (1, 384), 0), out=buf168)
        del arg178_1
        buf169 = reinterpret_tensor(buf168, (32, 64, 192), (12288, 192, 1), 0); del buf168  # reuse
        buf173 = reinterpret_tensor(buf142, (32, 64, 192), (12288, 192, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_344, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf169, buf150, buf161, arg173_1, arg179_1, arg180_1, arg181_1, buf173, 2048, 192, grid=grid(2048), stream=stream0)
        del arg173_1
        del arg179_1
        del arg180_1
        del arg181_1
        del buf150
        del buf161
        buf174 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg183_1, reinterpret_tensor(buf173, (2048, 192), (192, 1), 0), reinterpret_tensor(arg182_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf174)
        del arg182_1
        del arg183_1
        # Topologically Sorted Source Nodes: [x_345], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf175 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf174, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, False)
        del buf174
        buf176 = buf175[0]
        del buf175
        buf180 = reinterpret_tensor(buf173, (2048, 192), (192, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf176, (2048, 192), (192, 1), 0), reinterpret_tensor(arg184_1, (192, 192), (1, 192), 0), out=buf180)
        del arg184_1
        buf184 = reinterpret_tensor(buf176, (32, 64, 192), (12288, 192, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_349, layer_norm_33], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_33.run(buf169, buf180, arg185_1, arg186_1, arg187_1, buf184, 2048, 192, grid=grid(2048), stream=stream0)
        del arg186_1
        del arg187_1
        buf185 = reinterpret_tensor(buf167, (2048, 384), (384, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf184, (2048, 192), (192, 1), 0), reinterpret_tensor(arg188_1, (192, 384), (1, 192), 0), out=buf185)
        del arg188_1
        buf186 = reinterpret_tensor(buf185, (32, 64, 384), (24576, 384, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_351], Original ATen: [aten.silu]
        triton_poi_fused_silu_31.run(buf186, arg189_1, 786432, grid=grid(786432), stream=stream0)
        del arg189_1
        buf187 = reinterpret_tensor(buf184, (2048, 192), (192, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf186, (2048, 384), (384, 1), 0), reinterpret_tensor(arg190_1, (384, 192), (1, 384), 0), out=buf187)
        del arg190_1
        del buf186
        buf188 = reinterpret_tensor(buf187, (32, 64, 192), (12288, 192, 1), 0); del buf187  # reuse
        buf189 = empty_strided_cuda((32, 64, 1), (64, 1, 2048), torch.float32)
        buf190 = empty_strided_cuda((32, 64, 1), (64, 1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_349, x_355, x_356], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_35.run(buf188, buf169, buf180, arg185_1, arg191_1, buf189, buf190, 2048, 192, grid=grid(2048), stream=stream0)
        del arg185_1
        del arg191_1
        del buf169
        buf192 = reinterpret_tensor(buf180, (12288, 2, 8, 2), (32, 16, 2, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [x_359], Original ATen: [aten.clone]
        triton_poi_fused_clone_36.run(buf188, buf189, buf190, arg192_1, arg193_1, buf192, 4, 98304, grid=grid(4, 98304), stream=stream0)
        del arg192_1
        del arg193_1
        del buf189
        del buf190
        buf193 = reinterpret_tensor(buf188, (8, 192, 16, 16), (49152, 1, 3072, 192), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_37.run(buf192, buf193, 1536, 256, grid=grid(1536, 256), stream=stream0)
        del buf192
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 128, 16, 16), (32768, 1, 2048, 128))
        del arg194_1
        del buf193
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_26.run(buf195, arg195_1, arg196_1, arg197_1, arg198_1, 262144, grid=grid(262144), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        buf196 = empty_strided_cuda((8, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_38.run(buf107, buf195, buf196, 524288, grid=grid(524288), stream=stream0)
        del buf107
        buf197 = empty_strided_cuda((128, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [cat_4, x_363], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_39.run(arg199_1, buf197, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg199_1
        # Topologically Sorted Source Nodes: [cat_4, x_363], Original ATen: [aten.cat, aten.convolution]
        buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 128, 16, 16), (32768, 1, 2048, 128))
        del buf196
        del buf197
        buf199 = buf198; del buf198  # reuse
        buf200 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_364, x_365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_28.run(buf199, arg200_1, arg201_1, arg202_1, arg203_1, buf200, 262144, grid=grid(262144), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del buf199
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten.silu, aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 512, 16, 16), (131072, 1, 8192, 512))
        del arg204_1
        buf202 = buf201; del buf201  # reuse
        buf203 = empty_strided_cuda((8, 512, 16, 16), (131072, 1, 8192, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_367, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_40.run(buf202, arg205_1, arg206_1, arg207_1, arg208_1, buf203, 1048576, grid=grid(1048576), stream=stream0)
        del arg205_1
        del arg206_1
        del arg207_1
        del arg208_1
        del buf202
        # Topologically Sorted Source Nodes: [x_368, x_369], Original ATen: [aten.silu, aten.convolution]
        buf204 = extern_kernels.convolution(buf203, arg209_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf204, (8, 512, 8, 8), (32768, 1, 4096, 512))
        del arg209_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        buf206 = reinterpret_tensor(buf200, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_370, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_41.run(buf205, arg210_1, arg211_1, arg212_1, arg213_1, buf206, 262144, grid=grid(262144), stream=stream0)
        del arg210_1
        del arg211_1
        del arg212_1
        del arg213_1
        del buf205
        # Topologically Sorted Source Nodes: [x_371, x_372], Original ATen: [aten.silu, aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 160, 8, 8), (10240, 1, 1280, 160))
        del arg214_1
        del buf206
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_373], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf208, arg215_1, arg216_1, arg217_1, arg218_1, 81920, grid=grid(81920), stream=stream0)
        del arg215_1
        del arg216_1
        del arg217_1
        del arg218_1
        buf209 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_374], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_43.run(arg219_1, buf209, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del arg219_1
        # Topologically Sorted Source Nodes: [x_374], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 160, 8, 8), (10240, 1, 1280, 160))
        del buf209
        buf211 = buf210; del buf210  # reuse
        buf212 = empty_strided_cuda((8, 160, 8, 8), (10240, 1, 1280, 160), torch.float32)
        # Topologically Sorted Source Nodes: [x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_44.run(buf211, arg220_1, arg221_1, arg222_1, arg223_1, buf212, 81920, grid=grid(81920), stream=stream0)
        del arg220_1
        del arg221_1
        del arg222_1
        del arg223_1
        del buf211
        # Topologically Sorted Source Nodes: [x_376, x_377], Original ATen: [aten.silu, aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 240, 8, 8), (15360, 1, 1920, 240))
        del arg224_1
        del buf212
        buf217 = empty_strided_cuda((32, 16, 240), (3840, 240, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_35], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_45.run(buf213, arg225_1, arg226_1, buf217, 512, 240, grid=grid(512), stream=stream0)
        del arg225_1
        del arg226_1
        buf218 = empty_strided_cuda((512, 720), (720, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg228_1, reinterpret_tensor(buf217, (512, 240), (240, 1), 0), reinterpret_tensor(arg227_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf218)
        del arg227_1
        del arg228_1
        # Topologically Sorted Source Nodes: [x_380], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf219 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf218, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        buf220 = buf219[0]
        del buf219
        buf224 = reinterpret_tensor(buf217, (512, 240), (240, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf220, (512, 240), (240, 1), 0), reinterpret_tensor(arg229_1, (240, 240), (1, 240), 0), out=buf224)
        del arg229_1
        buf228 = reinterpret_tensor(buf220, (32, 16, 240), (3840, 240, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_384, layer_norm_36], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_46.run(buf213, buf224, arg230_1, arg231_1, arg232_1, buf228, 512, 240, grid=grid(512), stream=stream0)
        del arg231_1
        del arg232_1
        buf229 = empty_strided_cuda((512, 480), (480, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (512, 240), (240, 1), 0), reinterpret_tensor(arg233_1, (240, 480), (1, 240), 0), out=buf229)
        del arg233_1
        buf230 = reinterpret_tensor(buf229, (32, 16, 480), (7680, 480, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_386], Original ATen: [aten.silu]
        triton_poi_fused_silu_47.run(buf230, arg234_1, 245760, grid=grid(245760), stream=stream0)
        del arg234_1
        buf231 = reinterpret_tensor(buf228, (512, 240), (240, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (512, 480), (480, 1), 0), reinterpret_tensor(arg235_1, (480, 240), (1, 480), 0), out=buf231)
        del arg235_1
        buf232 = reinterpret_tensor(buf231, (32, 16, 240), (3840, 240, 1), 0); del buf231  # reuse
        buf236 = empty_strided_cuda((32, 16, 240), (3840, 240, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_384, x_390, layer_norm_37], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_48.run(buf232, buf213, buf224, arg230_1, arg236_1, arg237_1, arg238_1, buf236, 512, 240, grid=grid(512), stream=stream0)
        del arg230_1
        del arg236_1
        del arg237_1
        del arg238_1
        del buf213
        buf237 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg240_1, reinterpret_tensor(buf236, (512, 240), (240, 1), 0), reinterpret_tensor(arg239_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf237)
        del arg239_1
        del arg240_1
        # Topologically Sorted Source Nodes: [x_391], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf238 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf237, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf237, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf237, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        buf239 = buf238[0]
        del buf238
        buf243 = reinterpret_tensor(buf236, (512, 240), (240, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf239, (512, 240), (240, 1), 0), reinterpret_tensor(arg241_1, (240, 240), (1, 240), 0), out=buf243)
        del arg241_1
        buf247 = reinterpret_tensor(buf239, (32, 16, 240), (3840, 240, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_395, layer_norm_38], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_49.run(buf232, buf243, arg242_1, arg243_1, arg244_1, buf247, 512, 240, grid=grid(512), stream=stream0)
        del arg243_1
        del arg244_1
        buf248 = reinterpret_tensor(buf230, (512, 480), (480, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf247, (512, 240), (240, 1), 0), reinterpret_tensor(arg245_1, (240, 480), (1, 240), 0), out=buf248)
        del arg245_1
        buf249 = reinterpret_tensor(buf248, (32, 16, 480), (7680, 480, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [x_397], Original ATen: [aten.silu]
        triton_poi_fused_silu_47.run(buf249, arg246_1, 245760, grid=grid(245760), stream=stream0)
        del arg246_1
        buf250 = reinterpret_tensor(buf247, (512, 240), (240, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 480), (480, 1), 0), reinterpret_tensor(arg247_1, (480, 240), (1, 480), 0), out=buf250)
        del arg247_1
        buf251 = reinterpret_tensor(buf250, (32, 16, 240), (3840, 240, 1), 0); del buf250  # reuse
        buf255 = reinterpret_tensor(buf224, (32, 16, 240), (3840, 240, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_395, x_401, layer_norm_39], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_50.run(buf251, buf232, buf243, arg242_1, arg248_1, arg249_1, arg250_1, buf255, 512, 240, grid=grid(512), stream=stream0)
        del arg242_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf232
        del buf243
        buf256 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg252_1, reinterpret_tensor(buf255, (512, 240), (240, 1), 0), reinterpret_tensor(arg251_1, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf256)
        del arg251_1
        del arg252_1
        # Topologically Sorted Source Nodes: [x_402], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf257 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf256, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf256, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf256, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, False)
        del buf256
        buf258 = buf257[0]
        del buf257
        buf262 = reinterpret_tensor(buf255, (512, 240), (240, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf258, (512, 240), (240, 1), 0), reinterpret_tensor(arg253_1, (240, 240), (1, 240), 0), out=buf262)
        del arg253_1
        buf266 = reinterpret_tensor(buf258, (32, 16, 240), (3840, 240, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_406, layer_norm_40], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_49.run(buf251, buf262, arg254_1, arg255_1, arg256_1, buf266, 512, 240, grid=grid(512), stream=stream0)
        del arg255_1
        del arg256_1
        buf267 = reinterpret_tensor(buf249, (512, 480), (480, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf266, (512, 240), (240, 1), 0), reinterpret_tensor(arg257_1, (240, 480), (1, 240), 0), out=buf267)
        del arg257_1
        buf268 = reinterpret_tensor(buf267, (32, 16, 480), (7680, 480, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_408], Original ATen: [aten.silu]
        triton_poi_fused_silu_47.run(buf268, arg258_1, 245760, grid=grid(245760), stream=stream0)
        del arg258_1
        buf269 = reinterpret_tensor(buf266, (512, 240), (240, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 480), (480, 1), 0), reinterpret_tensor(arg259_1, (480, 240), (1, 480), 0), out=buf269)
        del arg259_1
        del buf268
        buf270 = reinterpret_tensor(buf269, (32, 16, 240), (3840, 240, 1), 0); del buf269  # reuse
        buf271 = empty_strided_cuda((32, 16, 1), (16, 1, 512), torch.float32)
        buf272 = empty_strided_cuda((32, 16, 1), (16, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_406, x_412, x_413], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_51.run(buf270, buf251, buf262, arg254_1, arg260_1, buf271, buf272, 512, 240, grid=grid(512), stream=stream0)
        del arg254_1
        del arg260_1
        del buf251
        buf274 = reinterpret_tensor(buf262, (7680, 2, 4, 2), (16, 8, 2, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_416], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf270, buf271, buf272, arg261_1, arg262_1, buf274, 4, 30720, grid=grid(4, 30720), stream=stream0)
        del arg261_1
        del arg262_1
        del buf271
        del buf272
        buf275 = reinterpret_tensor(buf270, (8, 240, 8, 8), (15360, 1, 1920, 240), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf274, buf275, 1920, 64, grid=grid(1920, 64), stream=stream0)
        del buf274
        # Topologically Sorted Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 160, 8, 8), (10240, 1, 1280, 160))
        del arg263_1
        del buf275
        buf277 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [x_418], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_42.run(buf277, arg264_1, arg265_1, arg266_1, arg267_1, 81920, grid=grid(81920), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        buf278 = empty_strided_cuda((8, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf208, buf277, buf278, 163840, grid=grid(163840), stream=stream0)
        del buf208
        buf279 = empty_strided_cuda((160, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5, x_420], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_55.run(arg268_1, buf279, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del arg268_1
        # Topologically Sorted Source Nodes: [cat_5, x_420], Original ATen: [aten.cat, aten.convolution]
        buf280 = extern_kernels.convolution(buf278, buf279, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 160, 8, 8), (10240, 1, 1280, 160))
        del buf278
        del buf279
        buf281 = buf280; del buf280  # reuse
        buf282 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [x_421, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.silu]
        triton_poi_fused__native_batch_norm_legit_no_training_silu_44.run(buf281, arg269_1, arg270_1, arg271_1, arg272_1, buf282, 81920, grid=grid(81920), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        del arg272_1
        del buf281
        # Topologically Sorted Source Nodes: [x_422, x_423], Original ATen: [aten.silu, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 640, 8, 8), (40960, 1, 5120, 640))
        del arg273_1
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_424], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf284, arg274_1, arg275_1, arg276_1, arg277_1, 327680, grid=grid(327680), stream=stream0)
        del arg274_1
        del arg275_1
        del arg276_1
        del arg277_1
        buf286 = empty_strided_cuda((8, 640, 1, 1), (640, 1, 5120, 5120), torch.float32)
        # Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten.silu, aten.mean]
        triton_per_fused_mean_silu_57.run(buf284, buf286, 5120, 64, grid=grid(5120), stream=stream0)
        del buf284
        buf287 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_429], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg279_1, reinterpret_tensor(buf286, (8, 640), (640, 1), 0), reinterpret_tensor(arg278_1, (640, 1000), (1, 640), 0), alpha=1, beta=1, out=buf287)
        del arg278_1
        del arg279_1
        del buf286
    return (buf287, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((1000, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
