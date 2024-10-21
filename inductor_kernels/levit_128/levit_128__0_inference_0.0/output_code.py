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


# kernel path: /tmp/torchinductor_sahanp/lj/clj3irzmjpv5kwpgx2gotcvfwvkd3p4k3oz7fdyq422agoaltijl.py
# Topologically Sorted Source Nodes: [input_40, input_41, conv2d_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_5 => convolution_5
#   input_40 => add_201, mul_238, mul_239, sub_78
#   input_41 => add_202, clamp_max_31, clamp_min_31, div_46, mul_240
# Graph fragment:
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_35), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_37), kwargs = {})
#   %add_201 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_39), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_201, 3), kwargs = {})
#   %clamp_min_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_202, 0), kwargs = {})
#   %clamp_max_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_31, 6), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_201, %clamp_max_31), kwargs = {})
#   %div_46 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_240, 6), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_46, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a4/ca4xow7d6k6mmkxetonxui4ydpiuv3xfr4soim7zmit4qbslxhy5.py
# Topologically Sorted Source Nodes: [input_42, input_43, conv2d_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_6 => convolution_6
#   input_42 => add_204, mul_242, mul_243, sub_79
#   input_43 => add_205, clamp_max_32, clamp_min_32, div_47, mul_244
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_43), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_242, %unsqueeze_45), kwargs = {})
#   %add_204 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_243, %unsqueeze_47), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_204, 3), kwargs = {})
#   %clamp_min_32 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_205, 0), kwargs = {})
#   %clamp_max_32 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_32, 6), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_204, %clamp_max_32), kwargs = {})
#   %div_47 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_244, 6), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_47, %arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/35/c35kvrrsc3gu6jf54n4pqwpnffwmulj5evlyrynpdxknkvsjbddk.py
# Topologically Sorted Source Nodes: [input_44, input_45, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_7 => convolution_7
#   input_44 => add_207, mul_246, mul_247, sub_80
#   input_45 => add_208, clamp_max_33, clamp_min_33, div_48, mul_248
# Graph fragment:
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_51), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_246, %unsqueeze_53), kwargs = {})
#   %add_207 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_247, %unsqueeze_55), kwargs = {})
#   %add_208 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_207, 3), kwargs = {})
#   %clamp_min_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_208, 0), kwargs = {})
#   %clamp_max_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_33, 6), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_207, %clamp_max_33), kwargs = {})
#   %div_48 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_248, 6), kwargs = {})
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%div_48, %arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/un/cunw35thfjoe27tdguhh62uxfk7vs6234mex5barmefmkuurxxpu.py
# Topologically Sorted Source Nodes: [input_46], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_46 => add_210, mul_250, mul_251, sub_81
# Graph fragment:
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_59), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_61), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_63), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ye/cyetlkystemcu24znv2i536cekddtoigrcw3nebjlf6k6ybra2p7.py
# Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_163 => clone_83
# Graph fragment:
#   %clone_83 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_117,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5u/c5upxjj6knxcy4ckemszhxocewygauiqrapezs2nf7g676xlimws.py
# Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_28 => clone_84
# Graph fragment:
#   %clone_84 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_56,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 196
    x2 = (xindex // 3136) % 4
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (50176*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y3/cy3etnahsremczdstudzgddrwour6ndkvlglqexrv6w5kbtbczfn.py
# Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_28 => clone_85
# Graph fragment:
#   %clone_85 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_57,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 4
    y2 = (yindex // 64)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (256*x3) + (50176*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4b/c4buywsrsfh3eu5jwasb7gmj6zcoqyxvzs752az2hdvlk7kn24jx.py
# Topologically Sorted Source Nodes: [getitem_3], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem_3 => index
# Graph fragment:
#   %index : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg26_1, [None, %arg27_1]), kwargs = {})
triton_poi_fused_index_7 = async_compile.triton('triton_poi_fused_index_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 153664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 38416
    x1 = (xindex // 38416)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 196, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 196)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 196")
    tmp6 = tl.load(in_ptr1 + (tmp4 + (196*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/md/cmdjtiy2sogw4loyr37pc6mmoqx2me7m4onichpmxeauyuzcg3e3.py
# Topologically Sorted Source Nodes: [mul_14, attn_28, attn_29], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_28 => add_213
#   attn_29 => amax_14, div_49, exp_14, sub_83, sum_15
#   mul_14 => mul_255
# Graph fragment:
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_359, 0.25), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_255, %index), kwargs = {})
#   %amax_14 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_213, [-1], True), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_213, %amax_14), kwargs = {})
#   %exp_14 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_83,), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_14, [-1], True), kwargs = {})
#   %div_49 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_14, %sum_15), kwargs = {})
triton_per_fused__softmax_add_mul_8 = async_compile.triton('triton_per_fused__softmax_add_mul_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_8(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x5 = xindex
    x0 = xindex % 784
    x3 = xindex % 196
    x6 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x5)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (196*x3) + (38432*x6)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fe/cfewpro4tulct3y6ubmn23o2u476xg3foc6ppi4gygbnd7k6q6oa.py
# Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_29 => clone_86
# Graph fragment:
#   %clone_86 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_59,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 4
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (256*x1) + (50176*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nr/cnrwc3ns7pstwl74eurbys74bejjdvv7s2ma47d3u3rds6zl6qjj.py
# Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   input_47 => add_214, clamp_max_34, clamp_min_34, div_50, mul_256
# Graph fragment:
#   %add_214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_363, 3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_214, 0), kwargs = {})
#   %clamp_max_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 6), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_363, %clamp_max_34), kwargs = {})
#   %div_50 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_256, 6), kwargs = {})
triton_poi_fused_hardswish_10 = async_compile.triton('triton_poi_fused_hardswish_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = (xindex // 128) % 196
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (6272*(x0 // 32)) + (25088*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xv/cxvv2vmp3xtsuvjyhpgj7kab3hzx734kpyrkwpkfvau6rfvewv4c.py
# Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten.add, aten.clone]
# Source node to ATen node mapping:
#   x_166 => add_217
#   x_167 => clone_88
# Graph fragment:
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_117, %view_367), kwargs = {})
#   %clone_88 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_217,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_11 = async_compile.triton('triton_poi_fused_add_clone_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oo/coo63xzvio57zt3ubnyhtosaq56ew3o4topgqbyos4fhluraf3dx.py
# Topologically Sorted Source Nodes: [batch_norm_70, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   batch_norm_70 => add_218, add_219, mul_260, mul_261, mul_262, reciprocal_70, sqrt_70, sub_85
#   x_169 => add_220, clamp_max_35, clamp_min_35, div_51, mul_263
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_370, %arg34_1), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg35_1, 1e-05), kwargs = {})
#   %sqrt_70 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_218,), kwargs = {})
#   %reciprocal_70 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_70,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_70, 1), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %mul_260), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_261, %arg36_1), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_262, %arg37_1), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_371, 3), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_220, 0), kwargs = {})
#   %clamp_max_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 6), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_371, %clamp_max_35), kwargs = {})
#   %div_51 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_263, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pt/cptsf3lpa2ozfy2otaiozlhogdykeu7fjcmr6jlt3maazoesprwm.py
# Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten.add, aten.clone]
# Source node to ATen node mapping:
#   x_173 => add_223
#   x_174 => clone_90
# Graph fragment:
#   %add_223 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_217, %view_375), kwargs = {})
#   %clone_90 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_223,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_13 = async_compile.triton('triton_poi_fused_add_clone_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pj/cpjv3o3q6buigvdtprv5tjgkjkdh7k4k3vmocm2kv4uhawerue72.py
# Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   input_55 => clone_112
# Graph fragment:
#   %clone_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_23,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 7
    x2 = (xindex // 896)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (3584*x2)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ms/cms2wvae3xilngztc4ddq6bra7vt2fxwi2um3d5tiyafx6vr52vw.py
# Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_36 => clone_113
# Graph fragment:
#   %clone_113 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_72,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (128*x1) + (6272*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/od/codhw5ny35kiwfqratj7fyq2zb42kaoumnggpbkklxq4yltoacmq.py
# Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_36 => clone_114
# Graph fragment:
#   %clone_114 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_73,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (640*x3) + (125440*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7u/c7ug6nexcbusnqagzithlwdfzs56iwprhfl5bmibyxqvdd32itzt.py
# Topologically Sorted Source Nodes: [getitem_19], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem_19 => index_4
# Graph fragment:
#   %index_4 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg119_1, [None, %arg120_1]), kwargs = {})
triton_poi_fused_index_17 = async_compile.triton('triton_poi_fused_index_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 76832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9604
    x1 = (xindex // 9604)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 196, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 196)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 196")
    tmp6 = tl.load(in_ptr1 + (tmp4 + (196*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/he/chemw2lnsw6arhidx34xm3kbl3ieyl354rmiisd3dfbnhd4pzsf4.py
# Topologically Sorted Source Nodes: [mul_18, attn_36, attn_37], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_36 => add_267
#   attn_37 => amax_18, div_61, exp_18, sub_104, sum_19
#   mul_18 => mul_318
# Graph fragment:
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_462, 0.25), kwargs = {})
#   %add_267 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %index_4), kwargs = {})
#   %amax_18 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_267, [-1], True), kwargs = {})
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_267, %amax_18), kwargs = {})
#   %exp_18 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_104,), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_18, [-1], True), kwargs = {})
#   %div_61 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_18, %sum_19), kwargs = {})
triton_per_fused__softmax_add_mul_18 = async_compile.triton('triton_per_fused__softmax_add_mul_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_18(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x5 = xindex
    x0 = xindex % 392
    x3 = xindex % 49
    x6 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x5)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (196*x3) + (9632*x6)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x4/cx4wj4zyotopubzdlaljuunv7hklrlflvckfvq4ejw4pnsy2lnv3.py
# Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_37 => clone_115
# Graph fragment:
#   %clone_115 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_75,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_poi_fused_clone_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 196
    x2 = (xindex // 12544) % 8
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (640*x1) + (125440*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nn/cnngdgw42bctdrthtqebwqfstvtoy5oq5zbjuvgwl4ccik7n6r3y.py
# Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   input_57 => add_268, clamp_max_42, clamp_min_42, div_62, mul_319
# Graph fragment:
#   %add_268 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_466, 3), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_268, 0), kwargs = {})
#   %clamp_max_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_466, %clamp_max_42), kwargs = {})
#   %div_62 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_319, 6), kwargs = {})
triton_poi_fused_hardswish_20 = async_compile.triton('triton_poi_fused_hardswish_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 512
    x1 = (xindex // 512) % 49
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (3136*(x0 // 64)) + (25088*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tt/cttpvcxwgqj7wvy5bgqudfqptpf7e5nosjipidkqtcav23giqrcx.py
# Topologically Sorted Source Nodes: [batch_norm_86], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_86 => add_269, add_270, mul_320, mul_321, mul_322, reciprocal_86, sqrt_86, sub_105
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_469, %arg122_1), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg123_1, 1e-05), kwargs = {})
#   %sqrt_86 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_269,), kwargs = {})
#   %reciprocal_86 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_86,), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_86, 1), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %mul_320), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_321, %arg124_1), kwargs = {})
#   %add_270 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_322, %arg125_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ez/cez5sdursiquwdw7vqhep3aeikk36pbtzfha2q3gylci4jnxzmhl.py
# Topologically Sorted Source Nodes: [batch_norm_87, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   batch_norm_87 => add_271, add_272, mul_323, mul_324, mul_325, reciprocal_87, sqrt_87, sub_106
#   x_215 => add_273, clamp_max_43, clamp_min_43, div_63, mul_326
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_473, %arg127_1), kwargs = {})
#   %add_271 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg128_1, 1e-05), kwargs = {})
#   %sqrt_87 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_271,), kwargs = {})
#   %reciprocal_87 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_87,), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_87, 1), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %mul_323), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_324, %arg129_1), kwargs = {})
#   %add_272 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_325, %arg130_1), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_474, 3), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_273, 0), kwargs = {})
#   %clamp_max_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 6), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_474, %clamp_max_43), kwargs = {})
#   %div_63 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_326, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yf/cyfu3g4lah6tbl22pvrltgrdqex7mh3ljiefp4trylkf7fxx3jdr.py
# Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_219 => add_276
# Graph fragment:
#   %add_276 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_470, %view_478), kwargs = {})
triton_poi_fused_add_23 = async_compile.triton('triton_poi_fused_add_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ef/cefzotrzip4ilppmfl73qgu27rqga5wmno6i3pcece63nz3svnfd.py
# Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_38 => clone_118
# Graph fragment:
#   %clone_118 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_76,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_24 = async_compile.triton('triton_poi_fused_clone_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (25088*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/px/cpxq33t2j6soog522by6342pi44qcj7gkubfxstugkij5dosrbmz.py
# Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_38 => clone_119
# Graph fragment:
#   %clone_119 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_77,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (512*x3) + (25088*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4y/c4yo7zxvcsjijx254kaaoiux76lnffnzktzdpmpcs2ffyc6hvd5s.py
# Topologically Sorted Source Nodes: [getitem_23], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem_23 => index_5
# Graph fragment:
#   %index_5 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg141_1, [None, %arg142_1]), kwargs = {})
triton_poi_fused_index_26 = async_compile.triton('triton_poi_fused_index_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2401
    x1 = (xindex // 2401)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 49, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 49)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 49")
    tmp6 = tl.load(in_ptr1 + (tmp4 + (49*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dz/cdz3elg2g3acmu2dbsh2muzc5i6wgym4fw3jqcokueigf6djbtfz.py
# Topologically Sorted Source Nodes: [mul_19, attn_38, attn_39], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_38 => add_279
#   attn_39 => amax_19, div_64, exp_19, sub_109, sum_20
#   mul_19 => mul_333
# Graph fragment:
#   %mul_333 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_486, 0.25), kwargs = {})
#   %add_279 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_333, %index_5), kwargs = {})
#   %amax_19 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_279, [-1], True), kwargs = {})
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_279, %amax_19), kwargs = {})
#   %exp_19 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_109,), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_19, [-1], True), kwargs = {})
#   %div_64 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_19, %sum_20), kwargs = {})
triton_per_fused__softmax_add_mul_27 = async_compile.triton('triton_per_fused__softmax_add_mul_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_27(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x5 = xindex
    x0 = xindex % 392
    x3 = xindex % 49
    x6 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x5)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (49*x3) + (2432*x6)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pu/cpuuspsjk4q3q54f2zhexmivcbyo4mrebd2iun4r5olsmhusfodz.py
# Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_39 => clone_120
# Graph fragment:
#   %clone_120 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_79,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (512*x1) + (25088*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4is7fjbb4pcqcporn4gmejvfzxilo6hzb44oxj6qgsrx75k5kod.py
# Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   input_59 => add_280, clamp_max_44, clamp_min_44, div_65, mul_334
# Graph fragment:
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_490, 3), kwargs = {})
#   %clamp_min_44 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_280, 0), kwargs = {})
#   %clamp_max_44 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_44, 6), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_490, %clamp_max_44), kwargs = {})
#   %div_65 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_334, 6), kwargs = {})
triton_poi_fused_hardswish_29 = async_compile.triton('triton_poi_fused_hardswish_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 49
    x2 = (xindex // 12544)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (1568*(x0 // 32)) + (12544*x2) + (x0 % 32)), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4v/c4vkqysmqajv6rhrg4m4g663fdexdt76dofrn2hhu3t6atyg6jlf.py
# Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   input_67 => clone_138
# Graph fragment:
#   %clone_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_30 = async_compile.triton('triton_poi_fused_clone_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 4
    x2 = (xindex // 1024) % 4
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (3584*x2) + (12544*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ah/cahpldpjziperzjemnxbfco53rbg6nvekpozbsicwdvtjou6ai3r.py
# Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_46 => clone_139
# Graph fragment:
#   %clone_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_92,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_31 = async_compile.triton('triton_poi_fused_clone_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (256*x1) + (4096*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0 + (16*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aw/cawqkn5i7qc2ijvk4at2ujwbkgm4tgizyqov5abecmfigsbmutp6.py
# Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_46 => clone_140
# Graph fragment:
#   %clone_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_93,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_32 = async_compile.triton('triton_poi_fused_clone_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (1280*x3) + (62720*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccytl22t4ztaztauhx25abbyca3ckfj6x62yoatm4gzsznude2dh.py
# Topologically Sorted Source Nodes: [getitem_39], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem_39 => index_9
# Graph fragment:
#   %index_9 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg234_1, [None, %arg235_1]), kwargs = {})
triton_poi_fused_index_33 = async_compile.triton('triton_poi_fused_index_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_33(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 784
    x1 = (xindex // 784)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 49, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 49)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 49")
    tmp6 = tl.load(in_ptr1 + (tmp4 + (49*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gy/cgyjeknw5fzth7kznahlxlfbyibrzsoqwaypq64svrdilydkylqp.py
# Topologically Sorted Source Nodes: [mul_23, attn_46, attn_47], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_46 => add_333
#   attn_47 => amax_23, div_76, exp_23, sub_130, sum_24
#   mul_23 => mul_396
# Graph fragment:
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_589, 0.25), kwargs = {})
#   %add_333 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_396, %index_9), kwargs = {})
#   %amax_23 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_333, [-1], True), kwargs = {})
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_333, %amax_23), kwargs = {})
#   %exp_23 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_130,), kwargs = {})
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_23, [-1], True), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_23, %sum_24), kwargs = {})
triton_per_fused__softmax_add_mul_34 = async_compile.triton('triton_per_fused__softmax_add_mul_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_34(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (49*x3)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/re/creoq5ebvjh3uqi3p3qouvx6h6ygamjanswfqsy4oml3t5pnqa5u.py
# Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_47 => clone_141
# Graph fragment:
#   %clone_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_95,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_35 = async_compile.triton('triton_poi_fused_clone_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 49
    x2 = (xindex // 3136) % 16
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (1280*x1) + (62720*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ui/cuinxx6626u7ftjwk4vneumkqtfkruihgfs4qts5lnvll4yxv3kl.py
# Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   input_69 => add_334, clamp_max_52, clamp_min_52, div_77, mul_397
# Graph fragment:
#   %add_334 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_593, 3), kwargs = {})
#   %clamp_min_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_334, 0), kwargs = {})
#   %clamp_max_52 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_52, 6), kwargs = {})
#   %mul_397 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_593, %clamp_max_52), kwargs = {})
#   %div_77 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_397, 6), kwargs = {})
triton_poi_fused_hardswish_36 = async_compile.triton('triton_poi_fused_hardswish_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 16
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (1024*(x0 // 64)) + (16384*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/45/c454dpe6gg5ue2qkrowpfkgwri3jnybd5elzxzcbujw6dh5ddu2p.py
# Topologically Sorted Source Nodes: [batch_norm_107], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_107 => add_335, add_336, mul_398, mul_399, mul_400, reciprocal_107, sqrt_107, sub_131
# Graph fragment:
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_596, %arg237_1), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg238_1, 1e-05), kwargs = {})
#   %sqrt_107 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_335,), kwargs = {})
#   %reciprocal_107 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_107,), kwargs = {})
#   %mul_398 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_107, 1), kwargs = {})
#   %mul_399 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_131, %mul_398), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_399, %arg239_1), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_400, %arg240_1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7s/c7sjco6f3k5rddazqev34aujtmyzx4i2kawzrxejdfbaqu7hnonp.py
# Topologically Sorted Source Nodes: [batch_norm_108, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   batch_norm_108 => add_337, add_338, mul_401, mul_402, mul_403, reciprocal_108, sqrt_108, sub_132
#   x_272 => add_339, clamp_max_53, clamp_min_53, div_78, mul_404
# Graph fragment:
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_600, %arg242_1), kwargs = {})
#   %add_337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg243_1, 1e-05), kwargs = {})
#   %sqrt_108 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_337,), kwargs = {})
#   %reciprocal_108 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_108,), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_108, 1), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %mul_401), kwargs = {})
#   %mul_403 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_402, %arg244_1), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_403, %arg245_1), kwargs = {})
#   %add_339 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_601, 3), kwargs = {})
#   %clamp_min_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_339, 0), kwargs = {})
#   %clamp_max_53 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_53, 6), kwargs = {})
#   %mul_404 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_601, %clamp_max_53), kwargs = {})
#   %div_78 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_404, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iu/ciucn2su2qxpabp2vrv5n7hkyt2fkssktzrxxpdunpf4ljsivqw7.py
# Topologically Sorted Source Nodes: [x_276], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_276 => add_342
# Graph fragment:
#   %add_342 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_597, %view_605), kwargs = {})
triton_poi_fused_add_39 = async_compile.triton('triton_poi_fused_add_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/md/cmd3svy2gjsvnlchxocqpurnuyfak2dk2rqewr7zisntl25jgfz7.py
# Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_144
# Graph fragment:
#   %clone_144 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_96,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_40 = async_compile.triton('triton_poi_fused_clone_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 12
    x3 = (xindex // 3072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4a/c4a6f5f5nghg3yvcoesgbddisbmtpnenuiyxfysibkzdm7lwwjqe.py
# Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_48 => clone_145
# Graph fragment:
#   %clone_145 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_97,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_41 = async_compile.triton('triton_poi_fused_clone_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 12
    y2 = (yindex // 192)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (768*x3) + (12288*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/u3/cu3baadygerhrckb4t4wz5edvsbwttalz6nrfgm7jxj7kpvktdm4.py
# Topologically Sorted Source Nodes: [getitem_43], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem_43 => index_10
# Graph fragment:
#   %index_10 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg256_1, [None, %arg257_1]), kwargs = {})
triton_poi_fused_index_42 = async_compile.triton('triton_poi_fused_index_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_42(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 16)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 16")
    tmp6 = tl.load(in_ptr1 + (tmp4 + (16*x1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4ttlmlyrkp5ld2sxvzo6cjjoxxxnjmpe64kfgsuxndwz7crvxx.py
# Topologically Sorted Source Nodes: [mul_24, attn_48, attn_49], Original ATen: [aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_48 => add_345
#   attn_49 => amax_24, div_79, exp_24, sub_135, sum_25
#   mul_24 => mul_411
# Graph fragment:
#   %mul_411 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_613, 0.25), kwargs = {})
#   %add_345 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_411, %index_10), kwargs = {})
#   %amax_24 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_345, [-1], True), kwargs = {})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_345, %amax_24), kwargs = {})
#   %exp_24 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_135,), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_24, [-1], True), kwargs = {})
#   %div_79 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_24, %sum_25), kwargs = {})
triton_per_fused__softmax_add_mul_43 = async_compile.triton('triton_per_fused__softmax_add_mul_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_mul_43(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + (r2 + (16*x3)), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r2 + (16*x0)), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (16*x3)), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/aa/caaahgx6hmi2f72naznwvn3iza43cmaaf7umepbehvf27uaqlmbf.py
# Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_49 => clone_146
# Graph fragment:
#   %clone_146 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_99,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_44 = async_compile.triton('triton_poi_fused_clone_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 12
    x3 = (xindex // 6144)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/st/cst4bkpc4qkzfl5pkmg7yspw2uynbraqz7vdkrrdnoswukfxmz3k.py
# Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.hardswish]
# Source node to ATen node mapping:
#   input_71 => add_346, clamp_max_54, clamp_min_54, div_80, mul_412
# Graph fragment:
#   %add_346 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_617, 3), kwargs = {})
#   %clamp_min_54 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_346, 0), kwargs = {})
#   %clamp_max_54 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_54, 6), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_617, %clamp_max_54), kwargs = {})
#   %div_80 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_412, 6), kwargs = {})
triton_poi_fused_hardswish_45 = async_compile.triton('triton_poi_fused_hardswish_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 384
    x1 = (xindex // 384) % 16
    x2 = (xindex // 6144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (512*(x0 // 32)) + (6144*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xf/cxfp72j7c6fgfkuylwvzcdtfsctvuozb7ati6afk5nbvhf525pt6.py
# Topologically Sorted Source Nodes: [x_320, x_321, batch_norm_126, batch_norm_127], Original ATen: [aten.add, aten.mean, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   batch_norm_126 => add_395, add_396, mul_468, mul_469, mul_470, reciprocal_126, sqrt_126, sub_154
#   batch_norm_127 => add_397, add_398, mul_471, mul_472, mul_473, reciprocal_127, sqrt_127, sub_155
#   x_320 => add_394
#   x_321 => mean_1
# Graph fragment:
#   %add_394 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_388, %view_701), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_394, [1]), kwargs = {})
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_1, %arg339_1), kwargs = {})
#   %add_395 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg340_1, 1e-05), kwargs = {})
#   %sqrt_126 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_395,), kwargs = {})
#   %reciprocal_126 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_126,), kwargs = {})
#   %mul_468 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_126, 1), kwargs = {})
#   %mul_469 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %mul_468), kwargs = {})
#   %mul_470 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_469, %arg341_1), kwargs = {})
#   %add_396 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_470, %arg342_1), kwargs = {})
#   %sub_155 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_1, %arg345_1), kwargs = {})
#   %add_397 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg346_1, 1e-05), kwargs = {})
#   %sqrt_127 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_397,), kwargs = {})
#   %reciprocal_127 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_127,), kwargs = {})
#   %mul_471 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_127, 1), kwargs = {})
#   %mul_472 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_155, %mul_471), kwargs = {})
#   %mul_473 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_472, %arg347_1), kwargs = {})
#   %add_398 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_473, %arg348_1), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_46 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (6144*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (6144*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 16.0
    tmp23 = tmp21 / tmp22
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 + tmp5
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tmp8 / tmp28
    tmp30 = tmp29 * tmp10
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp23 - tmp36
    tmp39 = tmp38 + tmp5
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tmp8 / tmp40
    tmp42 = tmp41 * tmp10
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(out_ptr1 + (x3), tmp35, xmask)
    tl.store(out_ptr2 + (x3), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qc/cqcyl7mingabr2mimkbxv6ujimxtfli6nbburbyaijnsegy6bqou.py
# Topologically Sorted Source Nodes: [add_81, x_323], Original ATen: [aten.add, aten.div]
# Source node to ATen node mapping:
#   add_81 => add_399
#   x_323 => div_91
# Graph fragment:
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %arg344_1), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg350_1), kwargs = {})
#   %add_399 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %add_tensor), kwargs = {})
#   %div_91 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_399, 2), kwargs = {})
triton_poi_fused_add_div_47 = async_compile.triton('triton_poi_fused_add_div_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (32, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (256, 128), (128, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (4, 196), (196, 1))
    assert_size_stride(arg27_1, (196, 196), (196, 1))
    assert_size_stride(arg28_1, (128, 128), (128, 1))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (256, 128), (128, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (128, 256), (256, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (256, 128), (128, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (4, 196), (196, 1))
    assert_size_stride(arg49_1, (196, 196), (196, 1))
    assert_size_stride(arg50_1, (128, 128), (128, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (256, 128), (128, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (128, 256), (256, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (256, 128), (128, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (4, 196), (196, 1))
    assert_size_stride(arg71_1, (196, 196), (196, 1))
    assert_size_stride(arg72_1, (128, 128), (128, 1))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (256, 128), (128, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (128, 256), (256, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (256, 128), (128, 1))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (4, 196), (196, 1))
    assert_size_stride(arg93_1, (196, 196), (196, 1))
    assert_size_stride(arg94_1, (128, 128), (128, 1))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (256, 128), (128, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, ), (1, ))
    assert_size_stride(arg104_1, (128, 256), (256, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (640, 128), (128, 1))
    assert_size_stride(arg110_1, (640, ), (1, ))
    assert_size_stride(arg111_1, (640, ), (1, ))
    assert_size_stride(arg112_1, (640, ), (1, ))
    assert_size_stride(arg113_1, (640, ), (1, ))
    assert_size_stride(arg114_1, (128, 128), (128, 1))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (8, 196), (196, 1))
    assert_size_stride(arg120_1, (49, 196), (196, 1))
    assert_size_stride(arg121_1, (256, 512), (512, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (512, 256), (256, 1))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (256, 512), (512, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (256, ), (1, ))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (512, 256), (256, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (8, 49), (49, 1))
    assert_size_stride(arg142_1, (49, 49), (49, 1))
    assert_size_stride(arg143_1, (256, 256), (256, 1))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (512, 256), (256, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, ), (1, ))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (256, 512), (512, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, ), (1, ))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (512, 256), (256, 1))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (8, 49), (49, 1))
    assert_size_stride(arg164_1, (49, 49), (49, 1))
    assert_size_stride(arg165_1, (256, 256), (256, 1))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (512, 256), (256, 1))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (256, 512), (512, 1))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (512, 256), (256, 1))
    assert_size_stride(arg181_1, (512, ), (1, ))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (8, 49), (49, 1))
    assert_size_stride(arg186_1, (49, 49), (49, 1))
    assert_size_stride(arg187_1, (256, 256), (256, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (512, 256), (256, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (256, 512), (512, 1))
    assert_size_stride(arg198_1, (256, ), (1, ))
    assert_size_stride(arg199_1, (256, ), (1, ))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (512, 256), (256, 1))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (8, 49), (49, 1))
    assert_size_stride(arg208_1, (49, 49), (49, 1))
    assert_size_stride(arg209_1, (256, 256), (256, 1))
    assert_size_stride(arg210_1, (256, ), (1, ))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (512, 256), (256, 1))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (512, ), (1, ))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (256, 512), (512, 1))
    assert_size_stride(arg220_1, (256, ), (1, ))
    assert_size_stride(arg221_1, (256, ), (1, ))
    assert_size_stride(arg222_1, (256, ), (1, ))
    assert_size_stride(arg223_1, (256, ), (1, ))
    assert_size_stride(arg224_1, (1280, 256), (256, 1))
    assert_size_stride(arg225_1, (1280, ), (1, ))
    assert_size_stride(arg226_1, (1280, ), (1, ))
    assert_size_stride(arg227_1, (1280, ), (1, ))
    assert_size_stride(arg228_1, (1280, ), (1, ))
    assert_size_stride(arg229_1, (256, 256), (256, 1))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (256, ), (1, ))
    assert_size_stride(arg232_1, (256, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (16, 49), (49, 1))
    assert_size_stride(arg235_1, (16, 49), (49, 1))
    assert_size_stride(arg236_1, (384, 1024), (1024, 1))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (768, 384), (384, 1))
    assert_size_stride(arg242_1, (768, ), (1, ))
    assert_size_stride(arg243_1, (768, ), (1, ))
    assert_size_stride(arg244_1, (768, ), (1, ))
    assert_size_stride(arg245_1, (768, ), (1, ))
    assert_size_stride(arg246_1, (384, 768), (768, 1))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (768, 384), (384, 1))
    assert_size_stride(arg252_1, (768, ), (1, ))
    assert_size_stride(arg253_1, (768, ), (1, ))
    assert_size_stride(arg254_1, (768, ), (1, ))
    assert_size_stride(arg255_1, (768, ), (1, ))
    assert_size_stride(arg256_1, (12, 16), (16, 1))
    assert_size_stride(arg257_1, (16, 16), (16, 1))
    assert_size_stride(arg258_1, (384, 384), (384, 1))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (768, 384), (384, 1))
    assert_size_stride(arg264_1, (768, ), (1, ))
    assert_size_stride(arg265_1, (768, ), (1, ))
    assert_size_stride(arg266_1, (768, ), (1, ))
    assert_size_stride(arg267_1, (768, ), (1, ))
    assert_size_stride(arg268_1, (384, 768), (768, 1))
    assert_size_stride(arg269_1, (384, ), (1, ))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (768, 384), (384, 1))
    assert_size_stride(arg274_1, (768, ), (1, ))
    assert_size_stride(arg275_1, (768, ), (1, ))
    assert_size_stride(arg276_1, (768, ), (1, ))
    assert_size_stride(arg277_1, (768, ), (1, ))
    assert_size_stride(arg278_1, (12, 16), (16, 1))
    assert_size_stride(arg279_1, (16, 16), (16, 1))
    assert_size_stride(arg280_1, (384, 384), (384, 1))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (384, ), (1, ))
    assert_size_stride(arg283_1, (384, ), (1, ))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (768, 384), (384, 1))
    assert_size_stride(arg286_1, (768, ), (1, ))
    assert_size_stride(arg287_1, (768, ), (1, ))
    assert_size_stride(arg288_1, (768, ), (1, ))
    assert_size_stride(arg289_1, (768, ), (1, ))
    assert_size_stride(arg290_1, (384, 768), (768, 1))
    assert_size_stride(arg291_1, (384, ), (1, ))
    assert_size_stride(arg292_1, (384, ), (1, ))
    assert_size_stride(arg293_1, (384, ), (1, ))
    assert_size_stride(arg294_1, (384, ), (1, ))
    assert_size_stride(arg295_1, (768, 384), (384, 1))
    assert_size_stride(arg296_1, (768, ), (1, ))
    assert_size_stride(arg297_1, (768, ), (1, ))
    assert_size_stride(arg298_1, (768, ), (1, ))
    assert_size_stride(arg299_1, (768, ), (1, ))
    assert_size_stride(arg300_1, (12, 16), (16, 1))
    assert_size_stride(arg301_1, (16, 16), (16, 1))
    assert_size_stride(arg302_1, (384, 384), (384, 1))
    assert_size_stride(arg303_1, (384, ), (1, ))
    assert_size_stride(arg304_1, (384, ), (1, ))
    assert_size_stride(arg305_1, (384, ), (1, ))
    assert_size_stride(arg306_1, (384, ), (1, ))
    assert_size_stride(arg307_1, (768, 384), (384, 1))
    assert_size_stride(arg308_1, (768, ), (1, ))
    assert_size_stride(arg309_1, (768, ), (1, ))
    assert_size_stride(arg310_1, (768, ), (1, ))
    assert_size_stride(arg311_1, (768, ), (1, ))
    assert_size_stride(arg312_1, (384, 768), (768, 1))
    assert_size_stride(arg313_1, (384, ), (1, ))
    assert_size_stride(arg314_1, (384, ), (1, ))
    assert_size_stride(arg315_1, (384, ), (1, ))
    assert_size_stride(arg316_1, (384, ), (1, ))
    assert_size_stride(arg317_1, (768, 384), (384, 1))
    assert_size_stride(arg318_1, (768, ), (1, ))
    assert_size_stride(arg319_1, (768, ), (1, ))
    assert_size_stride(arg320_1, (768, ), (1, ))
    assert_size_stride(arg321_1, (768, ), (1, ))
    assert_size_stride(arg322_1, (12, 16), (16, 1))
    assert_size_stride(arg323_1, (16, 16), (16, 1))
    assert_size_stride(arg324_1, (384, 384), (384, 1))
    assert_size_stride(arg325_1, (384, ), (1, ))
    assert_size_stride(arg326_1, (384, ), (1, ))
    assert_size_stride(arg327_1, (384, ), (1, ))
    assert_size_stride(arg328_1, (384, ), (1, ))
    assert_size_stride(arg329_1, (768, 384), (384, 1))
    assert_size_stride(arg330_1, (768, ), (1, ))
    assert_size_stride(arg331_1, (768, ), (1, ))
    assert_size_stride(arg332_1, (768, ), (1, ))
    assert_size_stride(arg333_1, (768, ), (1, ))
    assert_size_stride(arg334_1, (384, 768), (768, 1))
    assert_size_stride(arg335_1, (384, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (384, ), (1, ))
    assert_size_stride(arg338_1, (384, ), (1, ))
    assert_size_stride(arg339_1, (384, ), (1, ))
    assert_size_stride(arg340_1, (384, ), (1, ))
    assert_size_stride(arg341_1, (384, ), (1, ))
    assert_size_stride(arg342_1, (384, ), (1, ))
    assert_size_stride(arg343_1, (1000, 384), (384, 1))
    assert_size_stride(arg344_1, (1000, ), (1, ))
    assert_size_stride(arg345_1, (384, ), (1, ))
    assert_size_stride(arg346_1, (384, ), (1, ))
    assert_size_stride(arg347_1, (384, ), (1, ))
    assert_size_stride(arg348_1, (384, ), (1, ))
    assert_size_stride(arg349_1, (1000, 384), (384, 1))
    assert_size_stride(arg350_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg0_1
        del arg1_1
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((8, 16, 112, 112), (200704, 12544, 112, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40, input_41, conv2d_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_0.run(buf1, arg2_1, arg3_1, arg4_1, arg5_1, buf2, 1605632, grid=grid(1605632), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf1
        # Topologically Sorted Source Nodes: [input_41, conv2d_5], Original ATen: [aten.hardswish, aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg6_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg6_1
        del buf2
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((8, 32, 56, 56), (100352, 3136, 56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43, conv2d_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_1.run(buf4, arg7_1, arg8_1, arg9_1, arg10_1, buf5, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf4
        # Topologically Sorted Source Nodes: [input_43, conv2d_6], Original ATen: [aten.hardswish, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg11_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg11_1
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((8, 64, 28, 28), (50176, 784, 28, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45, conv2d_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardswish_2.run(buf7, arg12_1, arg13_1, arg14_1, arg15_1, buf8, 401408, grid=grid(401408), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [input_45, conv2d_7], Original ATen: [aten.hardswish, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg16_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg16_1
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, 200704, grid=grid(200704), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        buf11 = empty_strided_cuda((8, 196, 128), (25088, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf10, buf11, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf12 = reinterpret_tensor(buf8, (1568, 256), (256, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (1568, 128), (128, 1), 0), reinterpret_tensor(arg21_1, (128, 256), (1, 128), 0), out=buf12)
        del arg21_1
        buf13 = empty_strided_cuda((8, 4, 196, 16), (12544, 3136, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf13, 100352, grid=grid(100352), stream=stream0)
        buf14 = empty_strided_cuda((8, 4, 16, 196), (12544, 3136, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf14, 512, 196, grid=grid(512, 196), stream=stream0)
        buf15 = empty_strided_cuda((32, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf14, (32, 16, 196), (3136, 196, 1), 0), out=buf15)
        buf16 = empty_strided_cuda((4, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_3], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg27_1, arg26_1, buf16, 153664, grid=grid(153664), stream=stream0)
        del arg26_1
        del arg27_1
        buf19 = empty_strided_cuda((8, 4, 196, 196), (153728, 38432, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_14, attn_28, attn_29], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_8.run(buf15, buf16, buf19, 6272, 196, grid=grid(6272), stream=stream0)
        buf20 = reinterpret_tensor(buf11, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf20, 200704, grid=grid(200704), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf21 = empty_strided_cuda((32, 196, 32), (6272, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf19, (32, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf20, (32, 196, 32), (6272, 32, 1), 0), out=buf21)
        buf22 = reinterpret_tensor(buf20, (8, 196, 128), (25088, 128, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf21, buf22, 200704, grid=grid(200704), stream=stream0)
        buf23 = reinterpret_tensor(buf21, (1568, 128), (128, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (1568, 128), (128, 1), 0), reinterpret_tensor(arg28_1, (128, 128), (1, 128), 0), out=buf23)
        del arg28_1
        buf24 = reinterpret_tensor(buf23, (8, 196, 128), (25088, 128, 1), 0); del buf23  # reuse
        buf25 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_166, x_167], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_11.run(buf24, buf10, arg29_1, arg30_1, arg31_1, arg32_1, buf25, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        del arg32_1
        buf26 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1568, 128), (128, 1), 0), reinterpret_tensor(arg33_1, (128, 256), (1, 128), 0), out=buf26)
        del arg33_1
        buf27 = buf26; del buf26  # reuse
        buf28 = reinterpret_tensor(buf7, (8, 196, 256), (50176, 256, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_70, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf27, arg34_1, arg35_1, arg36_1, arg37_1, buf28, 401408, grid=grid(401408), stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        del arg37_1
        buf29 = reinterpret_tensor(buf25, (1568, 128), (128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (1568, 256), (256, 1), 0), reinterpret_tensor(arg38_1, (256, 128), (1, 256), 0), out=buf29)
        del arg38_1
        buf30 = buf24; del buf24  # reuse
        buf31 = reinterpret_tensor(buf10, (8, 196, 128), (25088, 128, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf30, buf29, arg39_1, arg40_1, arg41_1, arg42_1, buf31, 200704, grid=grid(200704), stream=stream0)
        del arg39_1
        del arg40_1
        del arg41_1
        del arg42_1
        buf32 = reinterpret_tensor(buf28, (1568, 256), (256, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1568, 128), (128, 1), 0), reinterpret_tensor(arg43_1, (128, 256), (1, 128), 0), out=buf32)
        del arg43_1
        buf33 = reinterpret_tensor(buf14, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf32, arg44_1, arg45_1, arg46_1, arg47_1, buf33, 100352, grid=grid(100352), stream=stream0)
        buf34 = reinterpret_tensor(buf13, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf32, arg44_1, arg45_1, arg46_1, arg47_1, buf34, 512, 196, grid=grid(512, 196), stream=stream0)
        buf35 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf34, (32, 16, 196), (3136, 196, 1), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_7], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg49_1, arg48_1, buf36, 153664, grid=grid(153664), stream=stream0)
        del arg48_1
        del arg49_1
        buf39 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [mul_15, attn_30, attn_31], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_8.run(buf35, buf36, buf39, 6272, 196, grid=grid(6272), stream=stream0)
        buf40 = reinterpret_tensor(buf31, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf32, arg44_1, arg45_1, arg46_1, arg47_1, buf40, 200704, grid=grid(200704), stream=stream0)
        del arg44_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf41 = reinterpret_tensor(buf29, (32, 196, 32), (6272, 32, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (32, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf40, (32, 196, 32), (6272, 32, 1), 0), out=buf41)
        buf42 = reinterpret_tensor(buf40, (8, 196, 128), (25088, 128, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf41, buf42, 200704, grid=grid(200704), stream=stream0)
        buf43 = reinterpret_tensor(buf41, (1568, 128), (128, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 128), (128, 1), 0), reinterpret_tensor(arg50_1, (128, 128), (1, 128), 0), out=buf43)
        del arg50_1
        buf44 = buf30; del buf30  # reuse
        buf45 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [x_177, x_178], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf44, buf43, arg51_1, arg52_1, arg53_1, arg54_1, buf45, 200704, grid=grid(200704), stream=stream0)
        del arg51_1
        del arg52_1
        del arg53_1
        del arg54_1
        buf46 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1568, 128), (128, 1), 0), reinterpret_tensor(arg55_1, (128, 256), (1, 128), 0), out=buf46)
        del arg55_1
        buf47 = buf46; del buf46  # reuse
        buf48 = reinterpret_tensor(buf27, (8, 196, 256), (50176, 256, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_74, x_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf47, arg56_1, arg57_1, arg58_1, arg59_1, buf48, 401408, grid=grid(401408), stream=stream0)
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        buf49 = reinterpret_tensor(buf45, (1568, 128), (128, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (1568, 256), (256, 1), 0), reinterpret_tensor(arg60_1, (256, 128), (1, 256), 0), out=buf49)
        del arg60_1
        buf50 = buf44; del buf44  # reuse
        buf51 = reinterpret_tensor(buf43, (8, 196, 128), (25088, 128, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_184, x_185], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf50, buf49, arg61_1, arg62_1, arg63_1, arg64_1, buf51, 200704, grid=grid(200704), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg64_1
        buf52 = reinterpret_tensor(buf48, (1568, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (1568, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 256), (1, 128), 0), out=buf52)
        del arg65_1
        buf53 = reinterpret_tensor(buf34, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf52, arg66_1, arg67_1, arg68_1, arg69_1, buf53, 100352, grid=grid(100352), stream=stream0)
        buf54 = reinterpret_tensor(buf33, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf52, arg66_1, arg67_1, arg68_1, arg69_1, buf54, 512, 196, grid=grid(512, 196), stream=stream0)
        buf55 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf54, (32, 16, 196), (3136, 196, 1), 0), out=buf55)
        buf56 = empty_strided_cuda((4, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_11], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg71_1, arg70_1, buf56, 153664, grid=grid(153664), stream=stream0)
        del arg70_1
        del arg71_1
        buf59 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [mul_16, attn_32, attn_33], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_8.run(buf55, buf56, buf59, 6272, 196, grid=grid(6272), stream=stream0)
        buf60 = reinterpret_tensor(buf51, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf52, arg66_1, arg67_1, arg68_1, arg69_1, buf60, 200704, grid=grid(200704), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        buf61 = reinterpret_tensor(buf49, (32, 196, 32), (6272, 32, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf59, (32, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf60, (32, 196, 32), (6272, 32, 1), 0), out=buf61)
        buf62 = reinterpret_tensor(buf60, (8, 196, 128), (25088, 128, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf61, buf62, 200704, grid=grid(200704), stream=stream0)
        buf63 = reinterpret_tensor(buf61, (1568, 128), (128, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (1568, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 128), (1, 128), 0), out=buf63)
        del arg72_1
        buf64 = buf50; del buf50  # reuse
        buf65 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_189], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf64, buf63, arg73_1, arg74_1, arg75_1, arg76_1, buf65, 200704, grid=grid(200704), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        buf66 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (1568, 128), (128, 1), 0), reinterpret_tensor(arg77_1, (128, 256), (1, 128), 0), out=buf66)
        del arg77_1
        buf67 = buf66; del buf66  # reuse
        buf68 = reinterpret_tensor(buf47, (8, 196, 256), (50176, 256, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_78, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf67, arg78_1, arg79_1, arg80_1, arg81_1, buf68, 401408, grid=grid(401408), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        buf69 = reinterpret_tensor(buf65, (1568, 128), (128, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (1568, 256), (256, 1), 0), reinterpret_tensor(arg82_1, (256, 128), (1, 256), 0), out=buf69)
        del arg82_1
        buf70 = buf64; del buf64  # reuse
        buf71 = reinterpret_tensor(buf63, (8, 196, 128), (25088, 128, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_195, x_196], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf70, buf69, arg83_1, arg84_1, arg85_1, arg86_1, buf71, 200704, grid=grid(200704), stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf72 = reinterpret_tensor(buf68, (1568, 256), (256, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_196], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (1568, 128), (128, 1), 0), reinterpret_tensor(arg87_1, (128, 256), (1, 128), 0), out=buf72)
        del arg87_1
        buf73 = reinterpret_tensor(buf54, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_5.run(buf72, arg88_1, arg89_1, arg90_1, arg91_1, buf73, 100352, grid=grid(100352), stream=stream0)
        buf74 = reinterpret_tensor(buf53, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_6.run(buf72, arg88_1, arg89_1, arg90_1, arg91_1, buf74, 512, 196, grid=grid(512, 196), stream=stream0)
        buf75 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf74, (32, 16, 196), (3136, 196, 1), 0), out=buf75)
        buf76 = empty_strided_cuda((4, 196, 196), (38416, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_15], Original ATen: [aten.index]
        triton_poi_fused_index_7.run(arg93_1, arg92_1, buf76, 153664, grid=grid(153664), stream=stream0)
        del arg92_1
        del arg93_1
        buf79 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [mul_17, attn_34, attn_35], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_8.run(buf75, buf76, buf79, 6272, 196, grid=grid(6272), stream=stream0)
        del buf75
        buf80 = reinterpret_tensor(buf71, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf72, arg88_1, arg89_1, arg90_1, arg91_1, buf80, 200704, grid=grid(200704), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        buf81 = reinterpret_tensor(buf69, (32, 196, 32), (6272, 32, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (32, 196, 196), (38432, 196, 1), 0), reinterpret_tensor(buf80, (32, 196, 32), (6272, 32, 1), 0), out=buf81)
        del buf79
        buf82 = reinterpret_tensor(buf80, (8, 196, 128), (25088, 128, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_10.run(buf81, buf82, 200704, grid=grid(200704), stream=stream0)
        buf83 = reinterpret_tensor(buf81, (1568, 128), (128, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1568, 128), (128, 1), 0), reinterpret_tensor(arg94_1, (128, 128), (1, 128), 0), out=buf83)
        del arg94_1
        buf84 = buf70; del buf70  # reuse
        buf85 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_199, x_200], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf84, buf83, arg95_1, arg96_1, arg97_1, arg98_1, buf85, 200704, grid=grid(200704), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        del arg98_1
        buf86 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1568, 128), (128, 1), 0), reinterpret_tensor(arg99_1, (128, 256), (1, 128), 0), out=buf86)
        del arg99_1
        buf87 = buf86; del buf86  # reuse
        buf88 = reinterpret_tensor(buf67, (8, 196, 256), (50176, 256, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_82, x_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12.run(buf87, arg100_1, arg101_1, arg102_1, arg103_1, buf88, 401408, grid=grid(401408), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg103_1
        del buf87
        buf89 = reinterpret_tensor(buf85, (1568, 128), (128, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_204], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1568, 256), (256, 1), 0), reinterpret_tensor(arg104_1, (256, 128), (1, 256), 0), out=buf89)
        del arg104_1
        buf90 = buf84; del buf84  # reuse
        buf91 = reinterpret_tensor(buf83, (8, 196, 128), (25088, 128, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_206, x_207], Original ATen: [aten.add, aten.clone]
        triton_poi_fused_add_clone_13.run(buf90, buf89, arg105_1, arg106_1, arg107_1, arg108_1, buf91, 200704, grid=grid(200704), stream=stream0)
        del arg105_1
        del arg106_1
        del arg107_1
        del arg108_1
        del buf89
        buf92 = empty_strided_cuda((1568, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_207], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (1568, 128), (128, 1), 0), reinterpret_tensor(arg109_1, (128, 640), (1, 128), 0), out=buf92)
        del arg109_1
        buf93 = empty_strided_cuda((8, 7, 7, 128), (6272, 896, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf90, buf93, 50176, grid=grid(50176), stream=stream0)
        buf94 = empty_strided_cuda((392, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (392, 128), (128, 1), 0), reinterpret_tensor(arg114_1, (128, 128), (1, 128), 0), out=buf94)
        del arg114_1
        buf95 = reinterpret_tensor(buf93, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf94, arg115_1, arg116_1, arg117_1, arg118_1, buf95, 50176, grid=grid(50176), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        del arg118_1
        buf96 = reinterpret_tensor(buf90, (8, 8, 16, 196), (25088, 3136, 196, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf92, arg110_1, arg111_1, arg112_1, arg113_1, buf96, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf97 = empty_strided_cuda((64, 49, 196), (9604, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf96, (64, 16, 196), (3136, 196, 1), 0), out=buf97)
        buf98 = empty_strided_cuda((8, 49, 196), (9604, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_19], Original ATen: [aten.index]
        triton_poi_fused_index_17.run(arg120_1, arg119_1, buf98, 76832, grid=grid(76832), stream=stream0)
        del arg119_1
        del arg120_1
        buf101 = empty_strided_cuda((8, 8, 49, 196), (77056, 9632, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_18, attn_36, attn_37], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_18.run(buf97, buf98, buf101, 3136, 196, grid=grid(3136), stream=stream0)
        del buf97
        buf102 = reinterpret_tensor(buf5, (8, 8, 196, 64), (100352, 12544, 64, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf92, arg110_1, arg111_1, arg112_1, arg113_1, buf102, 802816, grid=grid(802816), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg113_1
        del buf92
        buf103 = reinterpret_tensor(buf96, (64, 49, 64), (3136, 64, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (64, 49, 196), (9632, 196, 1), 0), reinterpret_tensor(buf102, (64, 196, 64), (12544, 64, 1), 0), out=buf103)
        del buf101
        del buf102
        buf104 = reinterpret_tensor(buf91, (8, 49, 512), (25088, 512, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_20.run(buf103, buf104, 200704, grid=grid(200704), stream=stream0)
        buf105 = reinterpret_tensor(buf74, (392, 256), (256, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (392, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 256), (1, 512), 0), out=buf105)
        del arg121_1
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_86], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf106, arg122_1, arg123_1, arg124_1, arg125_1, 100352, grid=grid(100352), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf107 = reinterpret_tensor(buf104, (392, 512), (512, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.mm]
        extern_kernels.mm(buf106, reinterpret_tensor(arg126_1, (256, 512), (1, 256), 0), out=buf107)
        del arg126_1
        buf108 = buf107; del buf107  # reuse
        buf109 = reinterpret_tensor(buf103, (8, 49, 512), (25088, 512, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_87, x_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf108, arg127_1, arg128_1, arg129_1, arg130_1, buf109, 200704, grid=grid(200704), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf110 = reinterpret_tensor(buf73, (392, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_217], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (392, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 256), (1, 512), 0), out=buf110)
        del arg131_1
        buf111 = reinterpret_tensor(buf106, (8, 49, 256), (12544, 256, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf111, buf110, arg132_1, arg133_1, arg134_1, arg135_1, 100352, grid=grid(100352), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf112 = reinterpret_tensor(buf109, (392, 512), (512, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (392, 256), (256, 1), 0), reinterpret_tensor(arg136_1, (256, 512), (1, 256), 0), out=buf112)
        del arg136_1
        buf113 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf112, arg137_1, arg138_1, arg139_1, arg140_1, buf113, 50176, grid=grid(50176), stream=stream0)
        buf114 = reinterpret_tensor(buf94, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf112, arg137_1, arg138_1, arg139_1, arg140_1, buf114, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf115 = empty_strided_cuda((64, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf113, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf114, (64, 16, 49), (784, 49, 1), 0), out=buf115)
        buf116 = empty_strided_cuda((8, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_23], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg142_1, arg141_1, buf116, 19208, grid=grid(19208), stream=stream0)
        del arg141_1
        del arg142_1
        buf119 = empty_strided_cuda((8, 8, 49, 49), (19456, 2432, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_19, attn_38, attn_39], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_27.run(buf115, buf116, buf119, 3136, 49, grid=grid(3136), stream=stream0)
        buf120 = reinterpret_tensor(buf110, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf112, arg137_1, arg138_1, arg139_1, arg140_1, buf120, 100352, grid=grid(100352), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf121 = empty_strided_cuda((64, 49, 32), (1568, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (64, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf120, (64, 49, 32), (1568, 32, 1), 0), out=buf121)
        buf122 = reinterpret_tensor(buf120, (8, 49, 256), (12544, 256, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf121, buf122, 100352, grid=grid(100352), stream=stream0)
        buf123 = reinterpret_tensor(buf121, (392, 256), (256, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (392, 256), (256, 1), 0), reinterpret_tensor(arg143_1, (256, 256), (1, 256), 0), out=buf123)
        del arg143_1
        buf124 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf124, buf123, arg144_1, arg145_1, arg146_1, arg147_1, 100352, grid=grid(100352), stream=stream0)
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        buf125 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (392, 256), (256, 1), 0), reinterpret_tensor(arg148_1, (256, 512), (1, 256), 0), out=buf125)
        del arg148_1
        buf126 = buf125; del buf125  # reuse
        buf127 = reinterpret_tensor(buf108, (8, 49, 512), (25088, 512, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_91, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf126, arg149_1, arg150_1, arg151_1, arg152_1, buf127, 200704, grid=grid(200704), stream=stream0)
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        buf128 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (392, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 256), (1, 512), 0), out=buf128)
        del arg153_1
        buf129 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf129, buf128, arg154_1, arg155_1, arg156_1, arg157_1, 100352, grid=grid(100352), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        buf130 = reinterpret_tensor(buf127, (392, 512), (512, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_231], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (392, 256), (256, 1), 0), reinterpret_tensor(arg158_1, (256, 512), (1, 256), 0), out=buf130)
        del arg158_1
        buf131 = reinterpret_tensor(buf114, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf130, arg159_1, arg160_1, arg161_1, arg162_1, buf131, 50176, grid=grid(50176), stream=stream0)
        buf132 = reinterpret_tensor(buf113, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf130, arg159_1, arg160_1, arg161_1, arg162_1, buf132, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf133 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf132, (64, 16, 49), (784, 49, 1), 0), out=buf133)
        buf134 = empty_strided_cuda((8, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_27], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg164_1, arg163_1, buf134, 19208, grid=grid(19208), stream=stream0)
        del arg163_1
        del arg164_1
        buf137 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [mul_20, attn_40, attn_41], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_27.run(buf133, buf134, buf137, 3136, 49, grid=grid(3136), stream=stream0)
        buf138 = reinterpret_tensor(buf128, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf130, arg159_1, arg160_1, arg161_1, arg162_1, buf138, 100352, grid=grid(100352), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del arg162_1
        buf139 = reinterpret_tensor(buf122, (64, 49, 32), (1568, 32, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (64, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf138, (64, 49, 32), (1568, 32, 1), 0), out=buf139)
        buf140 = reinterpret_tensor(buf138, (8, 49, 256), (12544, 256, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf139, buf140, 100352, grid=grid(100352), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (392, 256), (256, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_233], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (392, 256), (256, 1), 0), reinterpret_tensor(arg165_1, (256, 256), (1, 256), 0), out=buf141)
        del arg165_1
        buf142 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf142, buf141, arg166_1, arg167_1, arg168_1, arg169_1, 100352, grid=grid(100352), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        del arg169_1
        buf143 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (392, 256), (256, 1), 0), reinterpret_tensor(arg170_1, (256, 512), (1, 256), 0), out=buf143)
        del arg170_1
        buf144 = buf143; del buf143  # reuse
        buf145 = reinterpret_tensor(buf126, (8, 49, 512), (25088, 512, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_95, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf144, arg171_1, arg172_1, arg173_1, arg174_1, buf145, 200704, grid=grid(200704), stream=stream0)
        del arg171_1
        del arg172_1
        del arg173_1
        del arg174_1
        buf146 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [x_239], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (392, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 256), (1, 512), 0), out=buf146)
        del arg175_1
        buf147 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf147, buf146, arg176_1, arg177_1, arg178_1, arg179_1, 100352, grid=grid(100352), stream=stream0)
        del arg176_1
        del arg177_1
        del arg178_1
        del arg179_1
        buf148 = reinterpret_tensor(buf145, (392, 512), (512, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (392, 256), (256, 1), 0), reinterpret_tensor(arg180_1, (256, 512), (1, 256), 0), out=buf148)
        del arg180_1
        buf149 = reinterpret_tensor(buf132, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf132  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf148, arg181_1, arg182_1, arg183_1, arg184_1, buf149, 50176, grid=grid(50176), stream=stream0)
        buf150 = reinterpret_tensor(buf131, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf148, arg181_1, arg182_1, arg183_1, arg184_1, buf150, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf151 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf149, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf150, (64, 16, 49), (784, 49, 1), 0), out=buf151)
        buf152 = empty_strided_cuda((8, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_31], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg186_1, arg185_1, buf152, 19208, grid=grid(19208), stream=stream0)
        del arg185_1
        del arg186_1
        buf155 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [mul_21, attn_42, attn_43], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_27.run(buf151, buf152, buf155, 3136, 49, grid=grid(3136), stream=stream0)
        buf156 = reinterpret_tensor(buf146, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf148, arg181_1, arg182_1, arg183_1, arg184_1, buf156, 100352, grid=grid(100352), stream=stream0)
        del arg181_1
        del arg182_1
        del arg183_1
        del arg184_1
        buf157 = reinterpret_tensor(buf140, (64, 49, 32), (1568, 32, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf155, (64, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf156, (64, 49, 32), (1568, 32, 1), 0), out=buf157)
        buf158 = reinterpret_tensor(buf156, (8, 49, 256), (12544, 256, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf157, buf158, 100352, grid=grid(100352), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (392, 256), (256, 1), 0); del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_244], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (392, 256), (256, 1), 0), reinterpret_tensor(arg187_1, (256, 256), (1, 256), 0), out=buf159)
        del arg187_1
        buf160 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf160, buf159, arg188_1, arg189_1, arg190_1, arg191_1, 100352, grid=grid(100352), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        buf161 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (392, 256), (256, 1), 0), reinterpret_tensor(arg192_1, (256, 512), (1, 256), 0), out=buf161)
        del arg192_1
        buf162 = buf161; del buf161  # reuse
        buf163 = reinterpret_tensor(buf144, (8, 49, 512), (25088, 512, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_99, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf162, arg193_1, arg194_1, arg195_1, arg196_1, buf163, 200704, grid=grid(200704), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del arg196_1
        buf164 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (392, 512), (512, 1), 0), reinterpret_tensor(arg197_1, (512, 256), (1, 512), 0), out=buf164)
        del arg197_1
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf165, buf164, arg198_1, arg199_1, arg200_1, arg201_1, 100352, grid=grid(100352), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del arg201_1
        buf166 = reinterpret_tensor(buf163, (392, 512), (512, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (392, 256), (256, 1), 0), reinterpret_tensor(arg202_1, (256, 512), (1, 256), 0), out=buf166)
        del arg202_1
        buf167 = reinterpret_tensor(buf150, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf166, arg203_1, arg204_1, arg205_1, arg206_1, buf167, 50176, grid=grid(50176), stream=stream0)
        buf168 = reinterpret_tensor(buf149, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_25.run(buf166, arg203_1, arg204_1, arg205_1, arg206_1, buf168, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf169 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf167, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf168, (64, 16, 49), (784, 49, 1), 0), out=buf169)
        del buf167
        del buf168
        buf170 = empty_strided_cuda((8, 49, 49), (2401, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_35], Original ATen: [aten.index]
        triton_poi_fused_index_26.run(arg208_1, arg207_1, buf170, 19208, grid=grid(19208), stream=stream0)
        del arg207_1
        del arg208_1
        buf173 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [mul_22, attn_44, attn_45], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_27.run(buf169, buf170, buf173, 3136, 49, grid=grid(3136), stream=stream0)
        del buf169
        buf174 = reinterpret_tensor(buf164, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf166, arg203_1, arg204_1, arg205_1, arg206_1, buf174, 100352, grid=grid(100352), stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del arg206_1
        buf175 = reinterpret_tensor(buf158, (64, 49, 32), (1568, 32, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (64, 49, 49), (2432, 49, 1), 0), reinterpret_tensor(buf174, (64, 49, 32), (1568, 32, 1), 0), out=buf175)
        del buf173
        buf176 = reinterpret_tensor(buf174, (8, 49, 256), (12544, 256, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_29.run(buf175, buf176, 100352, grid=grid(100352), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (392, 256), (256, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (392, 256), (256, 1), 0), reinterpret_tensor(arg209_1, (256, 256), (1, 256), 0), out=buf177)
        del arg209_1
        del buf176
        buf178 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf178, buf177, arg210_1, arg211_1, arg212_1, arg213_1, 100352, grid=grid(100352), stream=stream0)
        del arg210_1
        del arg211_1
        del arg212_1
        del arg213_1
        buf179 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (392, 256), (256, 1), 0), reinterpret_tensor(arg214_1, (256, 512), (1, 256), 0), out=buf179)
        del arg214_1
        buf180 = buf179; del buf179  # reuse
        buf181 = reinterpret_tensor(buf162, (8, 49, 512), (25088, 512, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_103, x_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf180, arg215_1, arg216_1, arg217_1, arg218_1, buf181, 200704, grid=grid(200704), stream=stream0)
        del arg215_1
        del arg216_1
        del arg217_1
        del arg218_1
        del buf180
        buf182 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_261], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (392, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 256), (1, 512), 0), out=buf182)
        del arg219_1
        del buf181
        buf183 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_263], Original ATen: [aten.add]
        triton_poi_fused_add_23.run(buf183, buf182, arg220_1, arg221_1, arg222_1, arg223_1, 100352, grid=grid(100352), stream=stream0)
        del arg220_1
        del arg221_1
        del arg222_1
        del arg223_1
        buf184 = empty_strided_cuda((392, 1280), (1280, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (392, 256), (256, 1), 0), reinterpret_tensor(arg224_1, (256, 1280), (1, 256), 0), out=buf184)
        del arg224_1
        buf185 = empty_strided_cuda((8, 4, 4, 256), (4096, 1024, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf183, buf185, 32768, grid=grid(32768), stream=stream0)
        buf186 = empty_strided_cuda((128, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_267], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (128, 256), (256, 1), 0), reinterpret_tensor(arg229_1, (256, 256), (1, 256), 0), out=buf186)
        del arg229_1
        buf187 = reinterpret_tensor(buf185, (8, 16, 16, 16), (4096, 256, 16, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf186, arg230_1, arg231_1, arg232_1, arg233_1, buf187, 32768, grid=grid(32768), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        del buf186
        buf188 = reinterpret_tensor(buf183, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_32.run(buf184, arg225_1, arg226_1, arg227_1, arg228_1, buf188, 2048, 49, grid=grid(2048, 49), stream=stream0)
        buf189 = reinterpret_tensor(buf182, (128, 16, 49), (784, 49, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (128, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf188, (128, 16, 49), (784, 49, 1), 0), out=buf189)
        del buf187
        buf190 = empty_strided_cuda((16, 16, 49), (784, 49, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_39], Original ATen: [aten.index]
        triton_poi_fused_index_33.run(arg235_1, arg234_1, buf190, 12544, grid=grid(12544), stream=stream0)
        del arg234_1
        del arg235_1
        buf193 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [mul_23, attn_46, attn_47], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_34.run(buf189, buf190, buf193, 2048, 49, grid=grid(2048), stream=stream0)
        del buf189
        buf194 = reinterpret_tensor(buf88, (8, 16, 49, 64), (50176, 3136, 64, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_35.run(buf184, arg225_1, arg226_1, arg227_1, arg228_1, buf194, 401408, grid=grid(401408), stream=stream0)
        del arg225_1
        del arg226_1
        del arg227_1
        del arg228_1
        del buf184
        buf195 = empty_strided_cuda((128, 16, 64), (1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf193, (128, 16, 49), (784, 49, 1), 0), reinterpret_tensor(buf194, (128, 49, 64), (3136, 64, 1), 0), out=buf195)
        del buf193
        del buf194
        buf196 = empty_strided_cuda((8, 16, 1024), (16384, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_36.run(buf195, buf196, 131072, grid=grid(131072), stream=stream0)
        del buf195
        buf197 = empty_strided_cuda((128, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg236_1, (1024, 384), (1, 1024), 0), out=buf197)
        del arg236_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_107], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf198, arg237_1, arg238_1, arg239_1, arg240_1, 49152, grid=grid(49152), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        buf199 = empty_strided_cuda((128, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_270], Original ATen: [aten.mm]
        extern_kernels.mm(buf198, reinterpret_tensor(arg241_1, (384, 768), (1, 384), 0), out=buf199)
        del arg241_1
        buf200 = buf199; del buf199  # reuse
        buf201 = empty_strided_cuda((8, 16, 768), (12288, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_108, x_272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf200, arg242_1, arg243_1, arg244_1, arg245_1, buf201, 98304, grid=grid(98304), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        buf202 = empty_strided_cuda((128, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (128, 768), (768, 1), 0), reinterpret_tensor(arg246_1, (768, 384), (1, 768), 0), out=buf202)
        del arg246_1
        buf203 = reinterpret_tensor(buf198, (8, 16, 384), (6144, 384, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_276], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf203, buf202, arg247_1, arg248_1, arg249_1, arg250_1, 49152, grid=grid(49152), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf204 = reinterpret_tensor(buf201, (128, 768), (768, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (128, 384), (384, 1), 0), reinterpret_tensor(arg251_1, (384, 768), (1, 384), 0), out=buf204)
        del arg251_1
        buf205 = empty_strided_cuda((8, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf204, arg252_1, arg253_1, arg254_1, arg255_1, buf205, 24576, grid=grid(24576), stream=stream0)
        buf206 = empty_strided_cuda((8, 12, 16, 16), (3072, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf204, arg252_1, arg253_1, arg254_1, arg255_1, buf206, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf207 = empty_strided_cuda((96, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf206, (96, 16, 16), (256, 16, 1), 0), out=buf207)
        buf208 = empty_strided_cuda((12, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_43], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg257_1, arg256_1, buf208, 3072, grid=grid(3072), stream=stream0)
        del arg256_1
        del arg257_1
        buf211 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [mul_24, attn_48, attn_49], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_43.run(buf207, buf208, buf211, 1536, 16, grid=grid(1536), stream=stream0)
        buf212 = reinterpret_tensor(buf202, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf204, arg252_1, arg253_1, arg254_1, arg255_1, buf212, 49152, grid=grid(49152), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf213 = empty_strided_cuda((96, 16, 32), (512, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf211, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf212, (96, 16, 32), (512, 32, 1), 0), out=buf213)
        buf214 = reinterpret_tensor(buf212, (8, 16, 384), (6144, 384, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf213, buf214, 49152, grid=grid(49152), stream=stream0)
        buf215 = reinterpret_tensor(buf213, (128, 384), (384, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf214, (128, 384), (384, 1), 0), reinterpret_tensor(arg258_1, (384, 384), (1, 384), 0), out=buf215)
        del arg258_1
        buf216 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf216, buf215, arg259_1, arg260_1, arg261_1, arg262_1, 49152, grid=grid(49152), stream=stream0)
        del arg259_1
        del arg260_1
        del arg261_1
        del arg262_1
        buf217 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_281], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (128, 384), (384, 1), 0), reinterpret_tensor(arg263_1, (384, 768), (1, 384), 0), out=buf217)
        del arg263_1
        buf218 = buf217; del buf217  # reuse
        buf219 = reinterpret_tensor(buf200, (8, 16, 768), (12288, 768, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_112, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf218, arg264_1, arg265_1, arg266_1, arg267_1, buf219, 98304, grid=grid(98304), stream=stream0)
        del arg264_1
        del arg265_1
        del arg266_1
        del arg267_1
        buf220 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (128, 768), (768, 1), 0), reinterpret_tensor(arg268_1, (768, 384), (1, 768), 0), out=buf220)
        del arg268_1
        buf221 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf221, buf220, arg269_1, arg270_1, arg271_1, arg272_1, 49152, grid=grid(49152), stream=stream0)
        del arg269_1
        del arg270_1
        del arg271_1
        del arg272_1
        buf222 = reinterpret_tensor(buf219, (128, 768), (768, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (128, 384), (384, 1), 0), reinterpret_tensor(arg273_1, (384, 768), (1, 384), 0), out=buf222)
        del arg273_1
        buf223 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf222, arg274_1, arg275_1, arg276_1, arg277_1, buf223, 24576, grid=grid(24576), stream=stream0)
        buf224 = reinterpret_tensor(buf207, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf222, arg274_1, arg275_1, arg276_1, arg277_1, buf224, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf225 = reinterpret_tensor(buf205, (96, 16, 16), (256, 16, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [matmul_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf224, (96, 16, 16), (256, 16, 1), 0), out=buf225)
        buf226 = empty_strided_cuda((12, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_47], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg279_1, arg278_1, buf226, 3072, grid=grid(3072), stream=stream0)
        del arg278_1
        del arg279_1
        buf229 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [mul_25, attn_50, attn_51], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_43.run(buf225, buf226, buf229, 1536, 16, grid=grid(1536), stream=stream0)
        buf230 = reinterpret_tensor(buf220, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [matmul_51], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf222, arg274_1, arg275_1, arg276_1, arg277_1, buf230, 49152, grid=grid(49152), stream=stream0)
        del arg274_1
        del arg275_1
        del arg276_1
        del arg277_1
        buf231 = reinterpret_tensor(buf214, (96, 16, 32), (512, 32, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [matmul_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf230, (96, 16, 32), (512, 32, 1), 0), out=buf231)
        buf232 = reinterpret_tensor(buf230, (8, 16, 384), (6144, 384, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf231, buf232, 49152, grid=grid(49152), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (128, 384), (384, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_290], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (128, 384), (384, 1), 0), reinterpret_tensor(arg280_1, (384, 384), (1, 384), 0), out=buf233)
        del arg280_1
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf234, buf233, arg281_1, arg282_1, arg283_1, arg284_1, 49152, grid=grid(49152), stream=stream0)
        del arg281_1
        del arg282_1
        del arg283_1
        del arg284_1
        buf235 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_292], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (128, 384), (384, 1), 0), reinterpret_tensor(arg285_1, (384, 768), (1, 384), 0), out=buf235)
        del arg285_1
        buf236 = buf235; del buf235  # reuse
        buf237 = reinterpret_tensor(buf218, (8, 16, 768), (12288, 768, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_116, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf236, arg286_1, arg287_1, arg288_1, arg289_1, buf237, 98304, grid=grid(98304), stream=stream0)
        del arg286_1
        del arg287_1
        del arg288_1
        del arg289_1
        buf238 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_296], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (128, 768), (768, 1), 0), reinterpret_tensor(arg290_1, (768, 384), (1, 768), 0), out=buf238)
        del arg290_1
        buf239 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [x_298], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf239, buf238, arg291_1, arg292_1, arg293_1, arg294_1, 49152, grid=grid(49152), stream=stream0)
        del arg291_1
        del arg292_1
        del arg293_1
        del arg294_1
        buf240 = reinterpret_tensor(buf237, (128, 768), (768, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (128, 384), (384, 1), 0), reinterpret_tensor(arg295_1, (384, 768), (1, 384), 0), out=buf240)
        del arg295_1
        buf241 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf240, arg296_1, arg297_1, arg298_1, arg299_1, buf241, 24576, grid=grid(24576), stream=stream0)
        buf242 = reinterpret_tensor(buf225, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf225  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf240, arg296_1, arg297_1, arg298_1, arg299_1, buf242, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf243 = reinterpret_tensor(buf223, (96, 16, 16), (256, 16, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [matmul_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf241, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf242, (96, 16, 16), (256, 16, 1), 0), out=buf243)
        buf244 = empty_strided_cuda((12, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_51], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg301_1, arg300_1, buf244, 3072, grid=grid(3072), stream=stream0)
        del arg300_1
        del arg301_1
        buf247 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [mul_26, attn_52, attn_53], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_43.run(buf243, buf244, buf247, 1536, 16, grid=grid(1536), stream=stream0)
        buf248 = reinterpret_tensor(buf238, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [matmul_53], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf240, arg296_1, arg297_1, arg298_1, arg299_1, buf248, 49152, grid=grid(49152), stream=stream0)
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        buf249 = reinterpret_tensor(buf232, (96, 16, 32), (512, 32, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [matmul_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf247, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf248, (96, 16, 32), (512, 32, 1), 0), out=buf249)
        buf250 = reinterpret_tensor(buf248, (8, 16, 384), (6144, 384, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf249, buf250, 49152, grid=grid(49152), stream=stream0)
        buf251 = reinterpret_tensor(buf249, (128, 384), (384, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (128, 384), (384, 1), 0), reinterpret_tensor(arg302_1, (384, 384), (1, 384), 0), out=buf251)
        del arg302_1
        buf252 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [x_302], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf252, buf251, arg303_1, arg304_1, arg305_1, arg306_1, 49152, grid=grid(49152), stream=stream0)
        del arg303_1
        del arg304_1
        del arg305_1
        del arg306_1
        buf253 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_303], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (128, 384), (384, 1), 0), reinterpret_tensor(arg307_1, (384, 768), (1, 384), 0), out=buf253)
        del arg307_1
        buf254 = buf253; del buf253  # reuse
        buf255 = reinterpret_tensor(buf236, (8, 16, 768), (12288, 768, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_120, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf254, arg308_1, arg309_1, arg310_1, arg311_1, buf255, 98304, grid=grid(98304), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        del arg311_1
        buf256 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [x_307], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (128, 768), (768, 1), 0), reinterpret_tensor(arg312_1, (768, 384), (1, 768), 0), out=buf256)
        del arg312_1
        buf257 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [x_309], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf257, buf256, arg313_1, arg314_1, arg315_1, arg316_1, 49152, grid=grid(49152), stream=stream0)
        del arg313_1
        del arg314_1
        del arg315_1
        del arg316_1
        buf258 = reinterpret_tensor(buf255, (128, 768), (768, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (128, 384), (384, 1), 0), reinterpret_tensor(arg317_1, (384, 768), (1, 384), 0), out=buf258)
        del arg317_1
        buf259 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf258, arg318_1, arg319_1, arg320_1, arg321_1, buf259, 24576, grid=grid(24576), stream=stream0)
        buf260 = reinterpret_tensor(buf243, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf258, arg318_1, arg319_1, arg320_1, arg321_1, buf260, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf261 = reinterpret_tensor(buf241, (96, 16, 16), (256, 16, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [matmul_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf259, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf260, (96, 16, 16), (256, 16, 1), 0), out=buf261)
        del buf259
        buf262 = empty_strided_cuda((12, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem_55], Original ATen: [aten.index]
        triton_poi_fused_index_42.run(arg323_1, arg322_1, buf262, 3072, grid=grid(3072), stream=stream0)
        del arg322_1
        del arg323_1
        buf265 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [mul_27, attn_54, attn_55], Original ATen: [aten.mul, aten.add, aten._softmax]
        triton_per_fused__softmax_add_mul_43.run(buf261, buf262, buf265, 1536, 16, grid=grid(1536), stream=stream0)
        del buf261
        buf266 = reinterpret_tensor(buf256, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [matmul_55], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(buf258, arg318_1, arg319_1, arg320_1, arg321_1, buf266, 49152, grid=grid(49152), stream=stream0)
        del arg318_1
        del arg319_1
        del arg320_1
        del arg321_1
        buf267 = reinterpret_tensor(buf250, (96, 16, 32), (512, 32, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [matmul_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf266, (96, 16, 32), (512, 32, 1), 0), out=buf267)
        del buf265
        buf268 = reinterpret_tensor(buf266, (8, 16, 384), (6144, 384, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [input_77], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_45.run(buf267, buf268, 49152, grid=grid(49152), stream=stream0)
        buf269 = reinterpret_tensor(buf267, (128, 384), (384, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [x_312], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (128, 384), (384, 1), 0), reinterpret_tensor(arg324_1, (384, 384), (1, 384), 0), out=buf269)
        del arg324_1
        del buf268
        buf270 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_313], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf270, buf269, arg325_1, arg326_1, arg327_1, arg328_1, 49152, grid=grid(49152), stream=stream0)
        del arg325_1
        del arg326_1
        del arg327_1
        del arg328_1
        buf271 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_314], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (128, 384), (384, 1), 0), reinterpret_tensor(arg329_1, (384, 768), (1, 384), 0), out=buf271)
        del arg329_1
        buf272 = buf271; del buf271  # reuse
        buf273 = reinterpret_tensor(buf254, (8, 16, 768), (12288, 768, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_124, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf272, arg330_1, arg331_1, arg332_1, arg333_1, buf273, 98304, grid=grid(98304), stream=stream0)
        del arg330_1
        del arg331_1
        del arg332_1
        del arg333_1
        del buf272
        buf274 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_318], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 768), (768, 1), 0), reinterpret_tensor(arg334_1, (768, 384), (1, 768), 0), out=buf274)
        del arg334_1
        del buf273
        buf276 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        buf278 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_320, x_321, batch_norm_126, batch_norm_127], Original ATen: [aten.add, aten.mean, aten._native_batch_norm_legit_no_training]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_46.run(buf270, buf274, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg345_1, arg346_1, arg347_1, arg348_1, buf276, buf278, 3072, 16, grid=grid(3072), stream=stream0)
        del arg335_1
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        del arg341_1
        del arg342_1
        del arg345_1
        del arg346_1
        del arg347_1
        del arg348_1
        del buf270
        del buf274
        buf277 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_320, x_321, batch_norm_126], Original ATen: [aten.add, aten.mean, aten._native_batch_norm_legit_no_training]
        extern_kernels.mm(buf276, reinterpret_tensor(arg343_1, (384, 1000), (1, 384), 0), out=buf277)
        del arg343_1
        del buf276
        buf279 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_320, x_321, batch_norm_127], Original ATen: [aten.add, aten.mean, aten._native_batch_norm_legit_no_training]
        extern_kernels.mm(buf278, reinterpret_tensor(arg349_1, (384, 1000), (1, 384), 0), out=buf279)
        del arg349_1
        del buf278
        buf280 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [add_81, x_323], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_47.run(buf280, arg344_1, buf279, arg350_1, 8000, grid=grid(8000), stream=stream0)
        del arg344_1
        del arg350_1
        del buf279
    return (buf280, buf16, buf36, buf56, buf76, buf98, buf116, buf134, buf152, buf170, buf190, buf208, buf226, buf244, buf262, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg28_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg50_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg72_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg94_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((640, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((8, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((49, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    arg121_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg143_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg165_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg187_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1280, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    arg236_1 = rand_strided((384, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg258_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg280_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg302_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg324_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
