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


# kernel path: /tmp/torchinductor_sahanp/7p/c7pbsffdi6pg7gdxuiahtx7ra5cv5plqt3bi7tawwyz5urtmaksa.py
# Topologically Sorted Source Nodes: [x__6, x__7, layer_norm_44], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_44 => add_214, add_215, mul_258, mul_259, rsqrt_44, sub_94, var_mean_44
#   x__6 => cat_15
#   x__7 => add_181
# Graph fragment:
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_26, %permute_142], 1), kwargs = {})
#   %add_181 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_15, %arg4_1), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_181, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_181, %getitem_173), kwargs = {})
#   %add_214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_172, 1e-06), kwargs = {})
#   %rsqrt_44 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_214,), kwargs = {})
#   %mul_258 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %rsqrt_44), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_258, %arg9_1), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_259, %arg10_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 401
    x1 = (xindex // 401)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 401, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp36 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp47 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = x0
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tmp21 >= tmp22
        tmp24 = tl.full([1, 1], 1, tl.int64)
        tmp25 = tmp21 < tmp24
        tmp26 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp25 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp21 >= tmp24
        tmp28 = tl.full([1, 1], 401, tl.int64)
        tmp29 = tmp21 < tmp28
        tmp30 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp27 & xmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tmp30 + tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp27, tmp32, tmp33)
        tmp35 = tl.where(tmp25, tmp26, tmp34)
        tmp37 = tmp35 + tmp36
        tmp38 = tmp37 - tmp18
        tmp39 = 128.0
        tmp40 = tmp19 / tmp39
        tmp41 = 1e-06
        tmp42 = tmp40 + tmp41
        tmp43 = libdevice.rsqrt(tmp42)
        tmp44 = tmp38 * tmp43
        tmp46 = tmp44 * tmp45
        tmp48 = tmp46 + tmp47
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp48, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kf/ckfqsw47yknrabi5oylgurvyq75qjqwx3rbhb6r4ip2r5cspwxym.py
# Topologically Sorted Source Nodes: [x__6, x__7, x_171, layer_norm_45], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_45 => add_217, add_218, mul_260, mul_261, rsqrt_45, sub_95, var_mean_45
#   x_171 => add_216
#   x__6 => cat_15
#   x__7 => add_181
# Graph fragment:
#   %cat_15 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_26, %permute_142], 1), kwargs = {})
#   %add_181 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_15, %arg4_1), kwargs = {})
#   %add_216 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_181, %view_262), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_216, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_216, %getitem_182), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_181, 1e-06), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_217,), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_45), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_260, %arg15_1), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_261, %arg16_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_1 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 401
    x1 = (xindex // 401)
    x3 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 401, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((400*r2) + (51200*x1) + (((-1) + x0) % 400)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp19 = tmp17 + tmp18
        tmp20 = tmp16 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight, roffset == 0
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp20, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_out_ptr0 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp25 - tmp22
        tmp27 = 128.0
        tmp28 = tmp23 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uk/cuknfjne3veixpn5ww4lu2pvajv4pwrvubbskeavidwv5clhiqyt.py
# Topologically Sorted Source Nodes: [x_165, conv2d_3], Original ATen: [aten.floor, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten._unsafe_index, aten.clamp, aten.rsub, aten.convolution]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   x_165 => _unsafe_index_16, _unsafe_index_17, _unsafe_index_18, _unsafe_index_19, _unsafe_index_20, _unsafe_index_21, _unsafe_index_22, _unsafe_index_23, _unsafe_index_24, _unsafe_index_25, _unsafe_index_26, _unsafe_index_27, _unsafe_index_28, _unsafe_index_29, _unsafe_index_30, _unsafe_index_31, add_182, add_188, add_189, add_190, add_191, add_192, add_193, add_194, add_195, add_196, add_197, add_198, add_199, add_200, add_201, add_202, add_203, add_204, add_205, add_206, add_207, add_208, add_209, add_210, add_211, add_212, clamp_max_34, clamp_max_35, clamp_min_34, clamp_min_35, convert_element_type_5, floor_2, floor_3, iota_3, mul_212, mul_214, mul_215, mul_216, mul_217, mul_218, mul_219, mul_220, mul_221, mul_222, mul_223, mul_224, mul_225, mul_226, mul_227, mul_228, mul_229, mul_230, mul_231, mul_232, mul_233, mul_234, mul_235, mul_236, mul_237, mul_238, mul_239, mul_240, mul_241, mul_242, mul_243, mul_244, mul_245, mul_246, mul_247, mul_248, mul_249, mul_250, mul_251, mul_252, mul_253, mul_254, mul_255, mul_256, mul_257, sub_72, sub_74, sub_75, sub_78, sub_79, sub_80, sub_81, sub_82, sub_83, sub_84, sub_85, sub_86, sub_87, sub_88, sub_89, sub_90, sub_91, sub_92, sub_93
# Graph fragment:
#   %floor_3 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_1,), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (224,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_5, 0.5), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_182, 1.0714285714285714), kwargs = {})
#   %sub_72 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_212, 0.5), kwargs = {})
#   %floor_2 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub_72,), kwargs = {})
#   %_unsafe_index_16 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_36, %clamp_max_37]), kwargs = {})
#   %sub_75 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_72, %floor_2), kwargs = {})
#   %clamp_min_35 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_75, 0.0), kwargs = {})
#   %clamp_max_35 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_35, 1.0), kwargs = {})
#   %add_188 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_35, 1.0), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_188, -0.75), kwargs = {})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_214, -3.75), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %add_188), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, -6.0), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_189, %add_188), kwargs = {})
#   %sub_79 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_216, -3.0), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_16, %sub_79), kwargs = {})
#   %_unsafe_index_17 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_38, %clamp_max_39]), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_35, 1.25), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_217, 2.25), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %clamp_max_35), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %clamp_max_35), kwargs = {})
#   %add_190 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, 1), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_17, %add_190), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %mul_239), kwargs = {})
#   %_unsafe_index_18 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_40, %clamp_max_41]), kwargs = {})
#   %sub_81 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_35), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, 1.25), kwargs = {})
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_220, 2.25), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %sub_81), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_221, %sub_81), kwargs = {})
#   %add_191 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_222, 1), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_18, %add_191), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_198, %mul_240), kwargs = {})
#   %_unsafe_index_19 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_42, %clamp_max_43]), kwargs = {})
#   %sub_83 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_35), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, -0.75), kwargs = {})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_223, -3.75), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %sub_83), kwargs = {})
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_224, -6.0), kwargs = {})
#   %mul_225 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_192, %sub_83), kwargs = {})
#   %sub_85 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_225, -3.0), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_19, %sub_85), kwargs = {})
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_199, %mul_241), kwargs = {})
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_1, %floor_3), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_74, 0.0), kwargs = {})
#   %clamp_max_34 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 1.0), kwargs = {})
#   %add_193 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_34, 1.0), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_193, -0.75), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_226, -3.75), kwargs = {})
#   %mul_227 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %add_193), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_227, -6.0), kwargs = {})
#   %mul_228 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_194, %add_193), kwargs = {})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_228, -3.0), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_200, %sub_87), kwargs = {})
#   %_unsafe_index_20 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_44, %clamp_max_45]), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_20, %sub_79), kwargs = {})
#   %_unsafe_index_21 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_46, %clamp_max_47]), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_21, %add_190), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %mul_243), kwargs = {})
#   %_unsafe_index_22 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_48, %clamp_max_49]), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_22, %add_191), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_201, %mul_244), kwargs = {})
#   %_unsafe_index_23 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_50, %clamp_max_51]), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_23, %sub_85), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_202, %mul_245), kwargs = {})
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_34, 1.25), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_229, 2.25), kwargs = {})
#   %mul_230 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %clamp_max_34), kwargs = {})
#   %mul_231 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_230, %clamp_max_34), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_231, 1), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_203, %add_195), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_254, %mul_255), kwargs = {})
#   %_unsafe_index_24 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_52, %clamp_max_53]), kwargs = {})
#   %mul_246 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_24, %sub_79), kwargs = {})
#   %_unsafe_index_25 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_54, %clamp_max_55]), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_25, %add_190), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_246, %mul_247), kwargs = {})
#   %_unsafe_index_26 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_56, %clamp_max_57]), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_26, %add_191), kwargs = {})
#   %add_205 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_204, %mul_248), kwargs = {})
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_58, %clamp_max_59]), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_27, %sub_85), kwargs = {})
#   %add_206 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_205, %mul_249), kwargs = {})
#   %sub_89 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_34), kwargs = {})
#   %mul_232 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, 1.25), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_232, 2.25), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %sub_89), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_233, %sub_89), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_234, 1), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_206, %add_196), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_210, %mul_256), kwargs = {})
#   %_unsafe_index_28 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_60, %clamp_max_61]), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_28, %sub_79), kwargs = {})
#   %_unsafe_index_29 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_62, %clamp_max_63]), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_29, %add_190), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_250, %mul_251), kwargs = {})
#   %_unsafe_index_30 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_64, %clamp_max_65]), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_30, %add_191), kwargs = {})
#   %add_208 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_207, %mul_252), kwargs = {})
#   %_unsafe_index_31 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg0_1, [None, None, %clamp_max_66, %clamp_max_67]), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_31, %sub_85), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_208, %mul_253), kwargs = {})
#   %sub_91 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_34), kwargs = {})
#   %mul_235 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, -0.75), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_235, -3.75), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %sub_91), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_236, -6.0), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_197, %sub_91), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_237, -3.0), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_209, %sub_93), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_211, %mul_257), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_212, %arg5_1, %arg6_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr20': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2(in_ptr0, out_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 224) % 224
    x0 = xindex % 224
    x2 = (xindex // 50176)
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 239, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = x0
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 + tmp2
    tmp18 = tmp17 * tmp4
    tmp19 = tmp18 - tmp2
    tmp20 = libdevice.floor(tmp19)
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp21 - tmp9
    tmp23 = triton_helpers.maximum(tmp22, tmp11)
    tmp24 = triton_helpers.minimum(tmp23, tmp13)
    tmp25 = tl.load(in_ptr0 + (tmp24 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp19 - tmp20
    tmp27 = 0.0
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30 + tmp29
    tmp32 = -0.75
    tmp33 = tmp31 * tmp32
    tmp34 = -3.75
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp31
    tmp37 = -6.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 * tmp31
    tmp40 = -3.0
    tmp41 = tmp39 - tmp40
    tmp42 = tmp25 * tmp41
    tmp43 = triton_helpers.maximum(tmp21, tmp11)
    tmp44 = triton_helpers.minimum(tmp43, tmp13)
    tmp45 = tl.load(in_ptr0 + (tmp44 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp46 = 1.25
    tmp47 = tmp30 * tmp46
    tmp48 = 2.25
    tmp49 = tmp47 - tmp48
    tmp50 = tmp49 * tmp30
    tmp51 = tmp50 * tmp30
    tmp52 = tmp51 + tmp29
    tmp53 = tmp45 * tmp52
    tmp54 = tmp21 + tmp9
    tmp55 = triton_helpers.maximum(tmp54, tmp11)
    tmp56 = triton_helpers.minimum(tmp55, tmp13)
    tmp57 = tl.load(in_ptr0 + (tmp56 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp58 = tmp29 - tmp30
    tmp59 = tmp58 * tmp46
    tmp60 = tmp59 - tmp48
    tmp61 = tmp60 * tmp58
    tmp62 = tmp61 * tmp58
    tmp63 = tmp62 + tmp29
    tmp64 = tmp57 * tmp63
    tmp65 = tl.full([1], 2, tl.int64)
    tmp66 = tmp21 + tmp65
    tmp67 = triton_helpers.maximum(tmp66, tmp11)
    tmp68 = triton_helpers.minimum(tmp67, tmp13)
    tmp69 = tl.load(in_ptr0 + (tmp68 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp70 = 2.0
    tmp71 = tmp70 - tmp30
    tmp72 = tmp71 * tmp32
    tmp73 = tmp72 - tmp34
    tmp74 = tmp73 * tmp71
    tmp75 = tmp74 + tmp37
    tmp76 = tmp75 * tmp71
    tmp77 = tmp76 - tmp40
    tmp78 = tmp69 * tmp77
    tmp79 = triton_helpers.maximum(tmp8, tmp11)
    tmp80 = triton_helpers.minimum(tmp79, tmp13)
    tmp81 = tl.load(in_ptr0 + (tmp24 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp41
    tmp83 = tl.load(in_ptr0 + (tmp44 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp84 = tmp83 * tmp52
    tmp85 = tl.load(in_ptr0 + (tmp56 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp86 = tmp85 * tmp63
    tmp87 = tl.load(in_ptr0 + (tmp68 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp87 * tmp77
    tmp89 = tmp8 + tmp9
    tmp90 = triton_helpers.maximum(tmp89, tmp11)
    tmp91 = triton_helpers.minimum(tmp90, tmp13)
    tmp92 = tl.load(in_ptr0 + (tmp24 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp93 = tmp92 * tmp41
    tmp94 = tl.load(in_ptr0 + (tmp44 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp95 = tmp94 * tmp52
    tmp96 = tl.load(in_ptr0 + (tmp56 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp97 = tmp96 * tmp63
    tmp98 = tl.load(in_ptr0 + (tmp68 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp99 = tmp98 * tmp77
    tmp100 = tmp8 + tmp65
    tmp101 = triton_helpers.maximum(tmp100, tmp11)
    tmp102 = triton_helpers.minimum(tmp101, tmp13)
    tmp103 = tl.load(in_ptr0 + (tmp24 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp104 = tmp103 * tmp41
    tmp105 = tl.load(in_ptr0 + (tmp44 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp106 = tmp105 * tmp52
    tmp107 = tl.load(in_ptr0 + (tmp56 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp108 = tmp107 * tmp63
    tmp109 = tl.load(in_ptr0 + (tmp68 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp110 = tmp109 * tmp77
    tmp111 = tmp42 + tmp53
    tmp112 = tmp111 + tmp64
    tmp113 = tmp112 + tmp78
    tmp114 = tmp6 - tmp7
    tmp115 = triton_helpers.maximum(tmp114, tmp27)
    tmp116 = triton_helpers.minimum(tmp115, tmp29)
    tmp117 = tmp116 + tmp29
    tmp118 = tmp117 * tmp32
    tmp119 = tmp118 - tmp34
    tmp120 = tmp119 * tmp117
    tmp121 = tmp120 + tmp37
    tmp122 = tmp121 * tmp117
    tmp123 = tmp122 - tmp40
    tmp124 = tmp113 * tmp123
    tmp125 = tmp82 + tmp84
    tmp126 = tmp125 + tmp86
    tmp127 = tmp126 + tmp88
    tmp128 = tmp116 * tmp46
    tmp129 = tmp128 - tmp48
    tmp130 = tmp129 * tmp116
    tmp131 = tmp130 * tmp116
    tmp132 = tmp131 + tmp29
    tmp133 = tmp127 * tmp132
    tmp134 = tmp124 + tmp133
    tmp135 = tmp93 + tmp95
    tmp136 = tmp135 + tmp97
    tmp137 = tmp136 + tmp99
    tmp138 = tmp29 - tmp116
    tmp139 = tmp138 * tmp46
    tmp140 = tmp139 - tmp48
    tmp141 = tmp140 * tmp138
    tmp142 = tmp141 * tmp138
    tmp143 = tmp142 + tmp29
    tmp144 = tmp137 * tmp143
    tmp145 = tmp134 + tmp144
    tmp146 = tmp104 + tmp106
    tmp147 = tmp146 + tmp108
    tmp148 = tmp147 + tmp110
    tmp149 = tmp70 - tmp116
    tmp150 = tmp149 * tmp32
    tmp151 = tmp150 - tmp34
    tmp152 = tmp151 * tmp149
    tmp153 = tmp152 + tmp37
    tmp154 = tmp153 * tmp149
    tmp155 = tmp154 - tmp40
    tmp156 = tmp148 * tmp155
    tmp157 = tmp145 + tmp156
    tl.store(out_ptr20 + (x3), tmp157, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/su/csuyfxtkzcir75ek33qq2ly6nsl6vedoxextzdrnokdfnlpgi37y.py
# Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_46 => var_mean_46
#   x__10 => add_213
#   x__9 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_27, %permute_143], 1), kwargs = {})
#   %add_213 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_16, %arg8_1), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_213, [2]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_cat_native_layer_norm_3 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2) % 197
    x0 = xindex % 2
    x2 = (xindex // 394)
    x5 = xindex % 394
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp15 = tl.load(in_ptr3 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 197, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((196*r3) + (25088*x0) + (50176*x2) + (((-1) + x1) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp18, xmask)
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4m/c4mziy35za5dfb427w77pnyewc2kchtgqnpredkqipjkgcpawdeq.py
# Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_46 => var_mean_46
#   x__10 => add_213
#   x__9 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_27, %permute_143], 1), kwargs = {})
#   %add_213 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_16, %arg8_1), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_213, [2]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_cat_native_layer_norm_4 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tl/ctlq7fhq7d44fkyxmwvxx2n6smvvypy4r44vpiowcuj6ls7ypeco.py
# Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_46 => add_221, add_222, mul_265, mul_266, rsqrt_46, sub_96, var_mean_46
#   x__10 => add_213
#   x__9 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_27, %permute_143], 1), kwargs = {})
#   %add_213 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_16, %arg8_1), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_213, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_213, %getitem_184), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_183, 1e-06), kwargs = {})
#   %rsqrt_46 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_221,), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %rsqrt_46), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %arg21_1), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %arg22_1), kwargs = {})
triton_poi_fused_add_cat_native_layer_norm_5 = async_compile.triton('triton_poi_fused_add_cat_native_layer_norm_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 197
    x0 = xindex % 256
    x2 = (xindex // 50432)
    x4 = xindex % 50432
    x3 = (xindex // 256)
    x5 = xindex
    tmp15 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((196*x0) + (50176*x2) + (((-1) + x1) % 196)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp5, tmp13)
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 - tmp17
    tmp20 = 256.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-06
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp18 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr1 + (x5), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/as/casrqqyj6eopu2le2mkmoj6z24myqrf7mi53zuzgam3ihts6fgsh.py
# Topologically Sorted Source Nodes: [x__9, x__10, x_182, layer_norm_47], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_47 => add_224, add_225, mul_267, mul_268, rsqrt_47, sub_97, var_mean_47
#   x_182 => add_223
#   x__10 => add_213
#   x__9 => cat_16
# Graph fragment:
#   %cat_16 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_27, %permute_143], 1), kwargs = {})
#   %add_213 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_16, %arg8_1), kwargs = {})
#   %add_223 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_213, %view_272), kwargs = {})
#   %var_mean_47 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_223, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_97 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_223, %getitem_193), kwargs = {})
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-06), kwargs = {})
#   %rsqrt_47 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_224,), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %rsqrt_47), kwargs = {})
#   %mul_268 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %arg27_1), kwargs = {})
#   %add_225 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_268, %arg28_1), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_6 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr3 + (r2 + (256*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 197, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((196*r2) + (50176*x1) + (((-1) + x0) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp19 = tmp17 + tmp18
        tmp20 = tmp16 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight, roffset == 0
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp20, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_out_ptr0 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp25 - tmp22
        tmp27 = 256.0
        tmp28 = tmp23 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp34 = tmp32 * tmp33
        tmp36 = tmp34 + tmp35
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp36, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vn/cvnnwthasmukcasdiowdrbhv2ulp5g4s5vkjic7rmczg4royf4h3.py
# Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_184 => add_226, erf_25, mul_269, mul_270, mul_271
# Graph fragment:
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_274, 0.5), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_274, 0.7071067811865476), kwargs = {})
#   %erf_25 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_270,), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_25, 1), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %add_226), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/cikjqavfposgfjlmuksmtawovasl4xielua2fkiswtevx6pufzqv.py
# Topologically Sorted Source Nodes: [x_188, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_48 => add_228, add_229, mul_272, mul_273, rsqrt_48, sub_98, var_mean_48
#   x_188 => add_227
# Graph fragment:
#   %add_227 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, %view_276), kwargs = {})
#   %var_mean_48 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_227, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_227, %getitem_195), kwargs = {})
#   %add_228 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_194, 1e-06), kwargs = {})
#   %rsqrt_48 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_228,), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %rsqrt_48), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %arg33_1), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %arg34_1), kwargs = {})
triton_per_fused_add_native_layer_norm_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4r/c4r37dq62smqfe3sehxif64qj2qvabufgfnhetbhkhbghk3s7eei.py
# Topologically Sorted Source Nodes: [x_188, x_193, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => add_231, add_232, mul_274, mul_275, rsqrt_49, sub_99, var_mean_49
#   x_188 => add_227
#   x_193 => add_230
# Graph fragment:
#   %add_227 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_223, %view_276), kwargs = {})
#   %add_230 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_227, %view_282), kwargs = {})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_230, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_230, %getitem_204), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_203, 1e-06), kwargs = {})
#   %rsqrt_49 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_231,), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %rsqrt_49), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %arg39_1), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %arg40_1), kwargs = {})
triton_per_fused_add_native_layer_norm_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
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
    tmp5 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
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
    tmp14 = tl.full([1], 256, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp9 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tmp8 - tmp16
    tmp23 = 256.0
    tmp24 = tmp21 / tmp23
    tmp25 = 1e-06
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp8, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ev/cevyrmkeu5dhwphacobjeko6xzjarimf5g6i6npgahhkef44fg6b.py
# Topologically Sorted Source Nodes: [x_173], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_173 => add_219, erf_24, mul_262, mul_263, mul_264
# Graph fragment:
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_264, 0.5), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_264, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_263,), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %add_219), kwargs = {})
triton_poi_fused_gelu_10 = async_compile.triton('triton_poi_fused_gelu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1231872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/22/c22xlzxr66j5uktq45343tgfttl7zp3nmn6m34wlaifo32vmgbxs.py
# Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.native_layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   input_37 => add_242, add_243, clone_84, mul_286, mul_287, rsqrt_52, sub_102, var_mean_52
#   input_38 => add_244, erf_28, mul_288, mul_289, mul_290
# Graph fragment:
#   %clone_84 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_70,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_84, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_84, %getitem_217), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_216, 1e-06), kwargs = {})
#   %rsqrt_52 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_242,), kwargs = {})
#   %mul_286 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %rsqrt_52), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_286, %arg57_1), kwargs = {})
#   %add_243 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_287, %arg58_1), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, 0.5), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, 0.7071067811865476), kwargs = {})
#   %erf_28 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_289,), kwargs = {})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_28, 1), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_288, %add_244), kwargs = {})
triton_per_fused_gelu_native_layer_norm_11 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (51328*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (51328*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = 0.7071067811865476
    tmp35 = tmp31 * tmp34
    tmp36 = libdevice.erf(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp33 * tmp38
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp39, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4s/c4sdbvjr46a7xex6w62q2a3hnmn7m5njzjjopg4ciyi4sb4t7ecl.py
# Topologically Sorted Source Nodes: [tmp_12, layer_norm_54], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_54 => add_248, add_249, mul_296, mul_297, rsqrt_54, sub_104, var_mean_54
#   tmp_12 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_298, %slice_74], 1), kwargs = {})
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_17, %getitem_221), kwargs = {})
#   %add_248 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_220, 1e-06), kwargs = {})
#   %rsqrt_54 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_248,), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %rsqrt_54), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %arg65_1), kwargs = {})
#   %add_249 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %arg66_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_12 = async_compile.triton('triton_per_fused_cat_native_layer_norm_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp37 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 256, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 256.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp40, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/to/ctolkofzecmzsjdp4zrslnx5dvqpayzr5fr3zv7lhkm4bv4euv5p.py
# Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   linear_104 => add_250
# Graph fragment:
#   %add_250 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_302, %arg68_1), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4i7l6ama53rtytxcglph22apmnxeqqvuggijljfkyaqqmsuas66.py
# Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_12 => clone_86
# Graph fragment:
#   %clone_86 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (50432*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e2/ce22klqumuwvoipp4ctkodm5wprn4izfq3q4oxzn4opazdciqwrq.py
# Topologically Sorted Source Nodes: [attn_19], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_19 => div_6, exp_6, sum_7
# Graph fragment:
#   %mul_tensor_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_312, 1), kwargs = {})
#   %amax_default_5 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_10, [-1], True), kwargs = {})
#   %sub_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_7), kwargs = {})
triton_per_fused__softmax_15 = async_compile.triton('triton_per_fused__softmax_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_15(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (197*x0)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ou/cou6ynlsekfzahxrzqimegwwfsk3tpvdtl2yv2sm52f53pwuba4g.py
# Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_13 => clone_88
# Graph fragment:
#   %clone_88 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_31,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 197
    x2 = (xindex // 12608) % 4
    x3 = (xindex // 50432)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (50432*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y3/cy37tidtcnbmfvdudvv5yqngysyy2yc4yff4pvf6ezycw76qbjm5.py
# Topologically Sorted Source Nodes: [input_40, x_214, input_43, input_41, input_44], Original ATen: [aten.native_layer_norm, aten.add, aten.gelu]
# Source node to ATen node mapping:
#   input_40 => add_245, add_246, clone_85, mul_291, mul_292, rsqrt_53, sub_103, var_mean_53
#   input_41 => add_247, erf_29, mul_293, mul_294, mul_295
#   input_43 => add_252, add_253, mul_299, mul_300, rsqrt_55, sub_106, var_mean_55
#   input_44 => add_254, erf_30, mul_301, mul_302, mul_303
#   x_214 => add_251
# Graph fragment:
#   %clone_85 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%slice_72,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_53 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_85, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_251 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_76, %view_318), kwargs = {})
#   %var_mean_55 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_251, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_85, %getitem_219), kwargs = {})
#   %add_245 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_218, 1e-06), kwargs = {})
#   %rsqrt_53 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_245,), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %rsqrt_53), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %arg61_1), kwargs = {})
#   %add_246 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %arg62_1), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_246, 0.5), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_246, 0.7071067811865476), kwargs = {})
#   %erf_29 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_294,), kwargs = {})
#   %add_247 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_29, 1), kwargs = {})
#   %mul_295 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %add_247), kwargs = {})
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_251, %getitem_223), kwargs = {})
#   %add_252 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, 1e-06), kwargs = {})
#   %rsqrt_55 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_252,), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %rsqrt_55), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_299, %arg75_1), kwargs = {})
#   %add_253 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_300, %arg76_1), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_253, 0.5), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_253, 0.7071067811865476), kwargs = {})
#   %erf_30 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_302,), kwargs = {})
#   %add_254 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_30, 1), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %add_254), kwargs = {})
triton_per_fused_add_gelu_native_layer_norm_17 = async_compile.triton('triton_per_fused_add_gelu_native_layer_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr3': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 13, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_gelu_native_layer_norm_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr7, xnumel, rnumel):
    xnumel = 8
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
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (50432*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp35 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr7 + (r1), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr8 + (r1), None, eviction_policy='evict_last')
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
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp18 >= tmp18
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp18 < tmp20
    tmp22 = tl.load(in_ptr3 + (r1 + (256*x0)), tmp21, other=0.0)
    tmp23 = tmp18 >= tmp20
    tmp24 = tl.full([1], 197, tl.int64)
    tmp25 = tmp18 < tmp24
    tmp26 = tl.load(in_ptr0 + (256 + r1 + (256*(-1)) + (50432*x0)), tmp23, other=0.0)
    tmp27 = tl.load(in_ptr1 + (256 + r1 + (256*(-1)) + (50432*x0)), tmp23, other=0.0)
    tmp28 = tl.load(in_ptr2 + (tl.broadcast_to(r1, [RBLOCK])), tmp23, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp21, tmp22, tmp32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tmp4 - tmp12
    tmp39 = 256.0
    tmp40 = tmp17 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = 0.5
    tmp50 = tmp48 * tmp49
    tmp51 = 0.7071067811865476
    tmp52 = tmp48 * tmp51
    tmp53 = libdevice.erf(tmp52)
    tmp54 = 1.0
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 * tmp55
    tmp57 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp59 = tl.broadcast_to(tmp57, [RBLOCK])
    tmp61 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp62 = tmp61 / tmp11
    tmp63 = tmp57 - tmp62
    tmp64 = tmp63 * tmp63
    tmp65 = tl.broadcast_to(tmp64, [RBLOCK])
    tmp67 = triton_helpers.promote_to_tensor(tl.sum(tmp65, 0))
    tmp68 = tmp37 - tmp62
    tmp69 = tmp67 / tmp39
    tmp70 = tmp69 + tmp41
    tmp71 = libdevice.rsqrt(tmp70)
    tmp72 = tmp68 * tmp71
    tmp74 = tmp72 * tmp73
    tmp76 = tmp74 + tmp75
    tmp77 = tmp76 * tmp49
    tmp78 = tmp76 * tmp51
    tmp79 = libdevice.erf(tmp78)
    tmp80 = tmp79 + tmp54
    tmp81 = tmp77 * tmp80
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp37, None)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp56, None)
    tl.store(out_ptr7 + (r1 + (256*x0)), tmp81, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/de/cdewjazoyychc4kw3ibcrxrk5c3lwohsspovqg4xubwjfpvjemwz.py
# Topologically Sorted Source Nodes: [tmp_14, layer_norm_56, tmp_13, layer_norm_58], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_56 => add_255, add_256, mul_304, mul_305, rsqrt_56, sub_107, var_mean_56
#   layer_norm_58 => add_262, add_263, mul_312, mul_313, rsqrt_58, sub_110, var_mean_58
#   tmp_13 => cat_18
#   tmp_14 => cat_19
# Graph fragment:
#   %cat_19 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_300, %slice_83], 1), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_19, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %getitem_225), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_224, 1e-06), kwargs = {})
#   %rsqrt_56 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_255,), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %rsqrt_56), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_304, %arg79_1), kwargs = {})
#   %add_256 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_305, %arg80_1), kwargs = {})
#   %cat_18 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_320, %slice_81], 1), kwargs = {})
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_18, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_110 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_18, %getitem_229), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_228, 1e-06), kwargs = {})
#   %rsqrt_58 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_262,), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_110, %rsqrt_58), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_312, %arg93_1), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_313, %arg94_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_18 = async_compile.triton('triton_per_fused_cat_native_layer_norm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp56 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr8 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 401, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tl.load(in_ptr4 + (r2 + (128*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp4, tmp33, tmp15)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(xmask, tmp35, 0)
    tmp38 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp40 = tl.where(xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp41 / tmp25
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp49 = tmp16 - tmp26
    tmp50 = 128.0
    tmp51 = tmp32 / tmp50
    tmp52 = 1e-06
    tmp53 = tmp51 + tmp52
    tmp54 = libdevice.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp58 = tmp34 - tmp42
    tmp59 = tmp48 / tmp50
    tmp60 = tmp59 + tmp52
    tmp61 = libdevice.rsqrt(tmp60)
    tmp62 = tmp58 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp57 + tmp65
    tmp68 = tmp64 + tmp67
    tl.store(out_ptr6 + (r2 + (128*x3)), tmp66, xmask)
    tl.store(out_ptr7 + (r2 + (128*x3)), tmp68, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/df/cdffyn3oisjju6d5yj4b6y7c72mfwnngmuiwsrgkzt2po7gyx2pu.py
# Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   linear_109 => add_257
# Graph fragment:
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_322, %arg82_1), kwargs = {})
triton_poi_fused_add_19 = async_compile.triton('triton_poi_fused_add_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bf/cbf2yyym3uqmv6wgnrvuyhqedi7ut2icpigx4on5niwbjyxpsq6k.py
# Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_14 => clone_90
# Graph fragment:
#   %clone_90 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_33,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 401
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (51328*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (401*y3)), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cethpxqak2xi4m3h3zr4wlrirtcfk3lg3cnxobdrlwrynaeohyyw.py
# Topologically Sorted Source Nodes: [attn_22], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_22 => div_7, exp_7, sum_8
# Graph fragment:
#   %mul_tensor_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_332, 1), kwargs = {})
#   %amax_default_4 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_8, [-1], True), kwargs = {})
#   %sub_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_8, %amax_default_4), kwargs = {})
#   %mul_tensor_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_4, 0.1767766952966369), kwargs = {})
#   %exp_7 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_9,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [-1], True), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_7, %sum_8), kwargs = {})
triton_per_fused__softmax_21 = async_compile.triton('triton_per_fused__softmax_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_21(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 401
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 4
    x3 = (xindex // 4)
    tmp0 = tl.load(in_ptr0 + (r1 + (401*x0)), rmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp5, 0))
    tmp7 = tmp2 - tmp6
    tmp8 = 0.1767766952966369
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + (401*x2) + (1632*x3)), tmp15, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6c4yohbzma6vv76saahr422ab5p2xaas7572y6b6taakpnxzhir.py
# Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   matmul_15 => bmm_15
# Graph fragment:
#   %bmm_15 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_333, %view_334), kwargs = {})
triton_poi_fused_bmm_22 = async_compile.triton('triton_poi_fused_bmm_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 401
    x1 = (xindex // 401)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (401*(x1 % 4)) + (1632*(x1 // 4))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/33/c33bcqnfekze4x7nkcyjhcbsg2bvsxbrcp5dppfxouvzcarqfcdm.py
# Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_15 => clone_92
# Graph fragment:
#   %clone_92 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_35,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_23 = async_compile.triton('triton_poi_fused_clone_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 410624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 401
    x2 = (xindex // 12832) % 4
    x3 = (xindex // 51328)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (128*x1) + (51328*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (32*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/og/cogwg3afxidv74pyh5sfwvitx3wxb3ltjdlypvgsq5xxtbr44lda.py
# Topologically Sorted Source Nodes: [x_218, input_46, input_47], Original ATen: [aten.add, aten.native_layer_norm, aten.gelu]
# Source node to ATen node mapping:
#   input_46 => add_259, add_260, mul_307, mul_308, rsqrt_57, sub_109, var_mean_57
#   input_47 => add_261, erf_31, mul_309, mul_310, mul_311
#   x_218 => add_258
# Graph fragment:
#   %add_258 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_85, %view_338), kwargs = {})
#   %var_mean_57 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_258, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_258, %getitem_227), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_226, 1e-06), kwargs = {})
#   %rsqrt_57 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_259,), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %rsqrt_57), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %arg89_1), kwargs = {})
#   %add_260 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %arg90_1), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_260, 0.5), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_260, 0.7071067811865476), kwargs = {})
#   %erf_31 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_310,), kwargs = {})
#   %add_261 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_31, 1), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_309, %add_261), kwargs = {})
triton_per_fused_add_gelu_native_layer_norm_24 = async_compile.triton('triton_per_fused_add_gelu_native_layer_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_gelu_native_layer_norm_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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
    tmp16 = tl.load(in_out_ptr0 + (r1 + (128*x0)), xmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (r1 + (128*x0)), tmp3 & xmask, other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1, 1], 401, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (128 + r1 + (128*(-1)) + (51328*x0)), tmp5 & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (128 + r1 + (128*(-1)) + (51328*x0)), tmp5 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.where(tmp3, tmp4, tmp14)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp25 = tl.where(xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = tmp19 - tmp29
    tmp37 = 128.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-06
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = 0.5
    tmp48 = tmp46 * tmp47
    tmp49 = 0.7071067811865476
    tmp50 = tmp46 * tmp49
    tmp51 = libdevice.erf(tmp50)
    tmp52 = 1.0
    tmp53 = tmp51 + tmp52
    tmp54 = tmp48 * tmp53
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp19, xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp54, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yx/cyxb5q4vt622ytbykogqmzaayekkaee7d72fjn3drvqhasqxadwn.py
# Topologically Sorted Source Nodes: [tmp_13, x_223, layer_norm_59], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_59 => add_265, add_266, mul_314, mul_315, rsqrt_59, sub_111, var_mean_59
#   tmp_13 => cat_18
#   x_223 => add_264
# Graph fragment:
#   %cat_18 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_320, %slice_81], 1), kwargs = {})
#   %add_264 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_18, %view_346), kwargs = {})
#   %var_mean_59 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_264, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_264, %getitem_238), kwargs = {})
#   %add_265 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_237, 1e-06), kwargs = {})
#   %rsqrt_59 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_265,), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %rsqrt_59), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %arg99_1), kwargs = {})
#   %add_266 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %arg100_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_25 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (128*x3)), xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 401, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 128.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(in_out_ptr0 + (r2 + (128*x3)), tmp20, xmask)
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp47, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sx/csxn4e6w2g5xoql52n3dltlhbpcfb2xxqosadfrw5wmfhbmyravo.py
# Topologically Sorted Source Nodes: [tmp_15, x_234, layer_norm_61], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_61 => add_272, add_273, mul_321, mul_322, rsqrt_61, sub_113, var_mean_61
#   tmp_15 => cat_20
#   x_234 => add_271
# Graph fragment:
#   %cat_20 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_340, %slice_90], 1), kwargs = {})
#   %add_271 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_20, %view_356), kwargs = {})
#   %var_mean_61 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_271, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_271, %getitem_249), kwargs = {})
#   %add_272 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_248, 1e-06), kwargs = {})
#   %rsqrt_61 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_272,), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %rsqrt_61), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_321, %arg111_1), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_322, %arg112_1), kwargs = {})
triton_per_fused_add_cat_native_layer_norm_26 = async_compile.triton('triton_per_fused_add_cat_native_layer_norm_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (256*x3)), None)
    tmp18 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.full([1], 256, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp21 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp20 - tmp28
    tmp35 = 256.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tl.store(in_out_ptr0 + (r2 + (256*x3)), tmp20, None)
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp44, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/od/codvvs6d6oowxal2ieevpgmam5otykqnzam6k2vv3us6seclxphc.py
# Topologically Sorted Source Nodes: [tmp_22, layer_norm_84, tmp_21, x_323], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_84 => add_351, add_352, mul_412, mul_413, rsqrt_84, sub_139, var_mean_84
#   tmp_21 => cat_26
#   tmp_22 => cat_27
#   x_323 => add_358, mul_420, mul_421, rsqrt_86, sub_142, var_mean_86
# Graph fragment:
#   %cat_27 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_468, %slice_127], 1), kwargs = {})
#   %var_mean_84 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_27, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_139 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %getitem_337), kwargs = {})
#   %add_351 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_336, 1e-06), kwargs = {})
#   %rsqrt_84 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_351,), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_139, %rsqrt_84), kwargs = {})
#   %mul_413 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_412, %arg247_1), kwargs = {})
#   %add_352 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_413, %arg248_1), kwargs = {})
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_488, %slice_125], 1), kwargs = {})
#   %var_mean_86 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_26, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_142 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %getitem_341), kwargs = {})
#   %add_358 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_340, 1e-06), kwargs = {})
#   %rsqrt_86 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_358,), kwargs = {})
#   %mul_420 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_142, %rsqrt_86), kwargs = {})
#   %mul_421 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_420, %arg261_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_27 = async_compile.triton('triton_per_fused_cat_native_layer_norm_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp56 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr6 + (r2), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr7 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 401, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (128 + r2 + (128*((-1) + x0)) + (51328*x1)), tmp6 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tl.load(in_ptr4 + (r2 + (128*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp4, tmp33, tmp15)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(xmask, tmp35, 0)
    tmp38 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp40 = tl.where(xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp41 / tmp25
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp49 = tmp16 - tmp26
    tmp50 = 128.0
    tmp51 = tmp32 / tmp50
    tmp52 = 1e-06
    tmp53 = tmp51 + tmp52
    tmp54 = libdevice.rsqrt(tmp53)
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp58 = tmp34 - tmp42
    tmp59 = tmp48 / tmp50
    tmp60 = tmp59 + tmp52
    tmp61 = libdevice.rsqrt(tmp60)
    tmp62 = tmp58 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp57 + tmp65
    tl.store(out_ptr5 + (r2 + (128*x3)), tmp64, xmask)
    tl.store(out_ptr6 + (r2 + (128*x3)), tmp66, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ft/cftuxzs74vj75kkei2hv655ru735e3652qzqownr7iw2j7mbbg2z.py
# Topologically Sorted Source Nodes: [tmp_23, x_324], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   tmp_23 => cat_28
#   x_324 => add_360, mul_422, mul_423, rsqrt_87, sub_143, var_mean_87
# Graph fragment:
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_508, %slice_134], 1), kwargs = {})
#   %var_mean_87 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_28, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_143 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_28, %getitem_343), kwargs = {})
#   %add_360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_342, 1e-06), kwargs = {})
#   %rsqrt_87 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_360,), kwargs = {})
#   %mul_422 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_143, %rsqrt_87), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_422, %arg263_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_28 = async_compile.triton('triton_per_fused_cat_native_layer_norm_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp37 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (256*x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (256 + r2 + (256*((-1) + x0)) + (50432*x1)), tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = tl.full([1], 256, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp25 = tmp17 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 - tmp24
    tmp31 = 256.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp38, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zg/czgxfxltita2t4x7jlrtmeozyzvuypv4yomgncoe54hjgyvvxoxb.py
# Topologically Sorted Source Nodes: [dropout_102], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   dropout_102 => clone_138
# Graph fragment:
#   %clone_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_2,), kwargs = {})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (51328*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2g/c2gyga5nmnig34jjoveqhkoiyme6x4aws222ct6aiia4vp23ab4p.py
# Topologically Sorted Source Nodes: [dropout_103], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   dropout_103 => clone_139
# Graph fragment:
#   %clone_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%select_3,), kwargs = {})
triton_poi_fused_clone_30 = async_compile.triton('triton_poi_fused_clone_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50432*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qh/cqhecbv3vpl2p6efg2govrx3x5gvowu3eaizyxxdjndy7fnznlno.py
# Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_327 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_509, [0]), kwargs = {})
triton_poi_fused_mean_31 = async_compile.triton('triton_poi_fused_mean_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (8000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 2.0
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 240, 240), (172800, 57600, 240, 1))
    assert_size_stride(arg1_1, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg4_1, (1, 401, 128), (51328, 128, 1))
    assert_size_stride(arg5_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg6_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg8_1, (1, 197, 256), (50432, 256, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (384, 128), (128, 1))
    assert_size_stride(arg12_1, (384, ), (1, ))
    assert_size_stride(arg13_1, (128, 128), (128, 1))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (384, 128), (128, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (128, 384), (384, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (768, 256), (256, 1))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (256, 256), (256, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (768, 256), (256, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (256, 768), (768, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (768, 256), (256, 1))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (256, 256), (256, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (768, 256), (256, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (256, 768), (768, 1))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (768, 256), (256, 1))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (256, 256), (256, 1))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (768, 256), (256, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (256, 768), (768, 1))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (256, 128), (128, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (128, 256), (256, 1))
    assert_size_stride(arg64_1, (128, ), (1, ))
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
    assert_size_stride(arg77_1, (128, 256), (256, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128), (128, 1))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, 128), (128, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, 128), (128, 1))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, 128), (128, 1))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (256, 128), (128, 1))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (384, 128), (128, 1))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (128, 128), (128, 1))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (384, 128), (128, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (128, 384), (384, 1))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (768, 256), (256, 1))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (256, 256), (256, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (768, 256), (256, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (256, 768), (768, 1))
    assert_size_stride(arg116_1, (256, ), (1, ))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (768, 256), (256, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (256, 256), (256, 1))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (768, 256), (256, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (256, 768), (768, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (768, 256), (256, 1))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (256, 256), (256, 1))
    assert_size_stride(arg134_1, (256, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (768, 256), (256, 1))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (256, 768), (768, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (256, 128), (128, 1))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (128, 256), (256, 1))
    assert_size_stride(arg148_1, (128, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, 256), (256, 1))
    assert_size_stride(arg152_1, (256, ), (1, ))
    assert_size_stride(arg153_1, (256, 256), (256, 1))
    assert_size_stride(arg154_1, (256, ), (1, ))
    assert_size_stride(arg155_1, (256, 256), (256, 1))
    assert_size_stride(arg156_1, (256, ), (1, ))
    assert_size_stride(arg157_1, (256, 256), (256, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (128, 256), (256, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (128, 128), (128, 1))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, 128), (128, 1))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, 128), (128, 1))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, 128), (128, 1))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (256, 128), (128, 1))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (384, 128), (128, 1))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (128, 128), (128, 1))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (384, 128), (128, 1))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (128, 384), (384, 1))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (768, 256), (256, 1))
    assert_size_stride(arg192_1, (768, ), (1, ))
    assert_size_stride(arg193_1, (256, 256), (256, 1))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (768, 256), (256, 1))
    assert_size_stride(arg198_1, (768, ), (1, ))
    assert_size_stride(arg199_1, (256, 768), (768, 1))
    assert_size_stride(arg200_1, (256, ), (1, ))
    assert_size_stride(arg201_1, (256, ), (1, ))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (768, 256), (256, 1))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (256, 256), (256, 1))
    assert_size_stride(arg206_1, (256, ), (1, ))
    assert_size_stride(arg207_1, (256, ), (1, ))
    assert_size_stride(arg208_1, (256, ), (1, ))
    assert_size_stride(arg209_1, (768, 256), (256, 1))
    assert_size_stride(arg210_1, (768, ), (1, ))
    assert_size_stride(arg211_1, (256, 768), (768, 1))
    assert_size_stride(arg212_1, (256, ), (1, ))
    assert_size_stride(arg213_1, (256, ), (1, ))
    assert_size_stride(arg214_1, (256, ), (1, ))
    assert_size_stride(arg215_1, (768, 256), (256, 1))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (256, 256), (256, 1))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, ), (1, ))
    assert_size_stride(arg220_1, (256, ), (1, ))
    assert_size_stride(arg221_1, (768, 256), (256, 1))
    assert_size_stride(arg222_1, (768, ), (1, ))
    assert_size_stride(arg223_1, (256, 768), (768, 1))
    assert_size_stride(arg224_1, (256, ), (1, ))
    assert_size_stride(arg225_1, (128, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (256, 128), (128, 1))
    assert_size_stride(arg228_1, (256, ), (1, ))
    assert_size_stride(arg229_1, (256, ), (1, ))
    assert_size_stride(arg230_1, (256, ), (1, ))
    assert_size_stride(arg231_1, (128, 256), (256, 1))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (256, ), (1, ))
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, 256), (256, 1))
    assert_size_stride(arg236_1, (256, ), (1, ))
    assert_size_stride(arg237_1, (256, 256), (256, 1))
    assert_size_stride(arg238_1, (256, ), (1, ))
    assert_size_stride(arg239_1, (256, 256), (256, 1))
    assert_size_stride(arg240_1, (256, ), (1, ))
    assert_size_stride(arg241_1, (256, 256), (256, 1))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (256, ), (1, ))
    assert_size_stride(arg244_1, (256, ), (1, ))
    assert_size_stride(arg245_1, (128, 256), (256, 1))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, ), (1, ))
    assert_size_stride(arg249_1, (128, 128), (128, 1))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, 128), (128, 1))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, 128), (128, 1))
    assert_size_stride(arg254_1, (128, ), (1, ))
    assert_size_stride(arg255_1, (128, 128), (128, 1))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (256, 128), (128, 1))
    assert_size_stride(arg260_1, (256, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (1000, 128), (128, 1))
    assert_size_stride(arg266_1, (1000, ), (1, ))
    assert_size_stride(arg267_1, (1000, 256), (256, 1))
    assert_size_stride(arg268_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg0_1, arg1_1, stride=(12, 12), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 20, 20), (51200, 400, 20, 1))
        del arg1_1
        buf5 = empty_strided_cuda((8, 401, 128), (51328, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x__6, x__7, layer_norm_44], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_0.run(arg3_1, buf0, arg2_1, arg4_1, arg9_1, arg10_1, buf5, 3208, 128, grid=grid(3208), stream=stream0)
        del arg10_1
        del arg9_1
        buf6 = empty_strided_cuda((3208, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg12_1, reinterpret_tensor(buf5, (3208, 128), (128, 1), 0), reinterpret_tensor(arg11_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf6)
        del arg11_1
        del arg12_1
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf6, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf6, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf6, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf8 = buf7[0]
        del buf7
        buf12 = reinterpret_tensor(buf5, (3208, 128), (128, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf8, (3208, 128), (128, 1), 0), reinterpret_tensor(arg13_1, (128, 128), (1, 128), 0), out=buf12)
        del arg13_1
        buf13 = reinterpret_tensor(buf12, (8, 401, 128), (51328, 128, 1), 0); del buf12  # reuse
        buf96 = reinterpret_tensor(buf8, (8, 401, 128), (51328, 128, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x__6, x__7, x_171, layer_norm_45], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_1.run(buf13, arg3_1, buf0, arg2_1, arg4_1, arg14_1, arg15_1, arg16_1, buf96, 3208, 128, grid=grid(3208), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del arg2_1
        del arg3_1
        del arg4_1
        del buf0
        buf37 = empty_strided_cuda((8, 3, 224, 224), (150528, 50176, 224, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_165, conv2d_3], Original ATen: [aten.floor, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten._unsafe_index, aten.clamp, aten.rsub, aten.convolution]
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2.run(arg0_1, buf37, 1204224, grid=grid(1204224), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_165, conv2d_3], Original ATen: [aten.add, aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg5_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 14, 14), (50176, 196, 14, 1))
        del arg5_1
        del buf37
        buf39 = empty_strided_cuda((8, 197, 1, 2), (394, 2, 3168, 1), torch.float32)
        buf40 = empty_strided_cuda((8, 197, 1, 2), (394, 2, 3168, 1), torch.float32)
        buf41 = empty_strided_cuda((8, 197, 1, 2), (394, 2, 3168, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_3.run(arg7_1, buf38, arg6_1, arg8_1, buf39, buf40, buf41, 3152, 128, grid=grid(3152), stream=stream0)
        buf42 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        buf43 = empty_strided_cuda((8, 197, 1), (197, 1, 1600), torch.float32)
        # Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_4.run(buf39, buf40, buf41, buf42, buf43, 1576, 2, grid=grid(1576), stream=stream0)
        del buf39
        del buf40
        del buf41
        buf46 = empty_strided_cuda((8, 197, 256), (50432, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x__9, x__10, layer_norm_46], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_poi_fused_add_cat_native_layer_norm_5.run(arg7_1, buf38, arg6_1, arg8_1, buf42, buf43, arg21_1, arg22_1, buf46, 403456, grid=grid(403456), stream=stream0)
        del arg21_1
        del arg22_1
        del buf42
        del buf43
        buf47 = empty_strided_cuda((1576, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg24_1, reinterpret_tensor(buf46, (1576, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf47)
        del arg23_1
        del arg24_1
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf48 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf47, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf47, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf49 = buf48[0]
        del buf48
        buf53 = reinterpret_tensor(buf46, (1576, 256), (256, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf49, (1576, 256), (256, 1), 0), reinterpret_tensor(arg25_1, (256, 256), (1, 256), 0), out=buf53)
        del arg25_1
        buf54 = reinterpret_tensor(buf53, (8, 197, 256), (50432, 256, 1), 0); del buf53  # reuse
        buf58 = reinterpret_tensor(buf49, (8, 197, 256), (50432, 256, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [x__9, x__10, x_182, layer_norm_47], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_red_fused_add_cat_native_layer_norm_6.run(buf54, arg7_1, buf38, arg6_1, arg8_1, arg26_1, arg27_1, arg28_1, buf58, 1576, 256, grid=grid(1576), stream=stream0)
        del arg26_1
        del arg27_1
        del arg28_1
        del arg6_1
        del arg7_1
        del arg8_1
        del buf38
        buf59 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf58, (1576, 256), (256, 1), 0), reinterpret_tensor(arg29_1, (256, 768), (1, 256), 0), out=buf59)
        del arg29_1
        buf60 = reinterpret_tensor(buf59, (8, 197, 768), (151296, 768, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf60, arg30_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg30_1
        buf61 = reinterpret_tensor(buf58, (1576, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf60, (1576, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 256), (1, 768), 0), out=buf61)
        del arg31_1
        buf65 = empty_strided_cuda((8, 197, 256), (50432, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_188, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf54, buf61, arg32_1, arg33_1, arg34_1, buf65, 1576, 256, grid=grid(1576), stream=stream0)
        del arg33_1
        del arg34_1
        buf66 = reinterpret_tensor(buf60, (1576, 768), (768, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg36_1, reinterpret_tensor(buf65, (1576, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf66)
        del arg35_1
        del arg36_1
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf67 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf66, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf66, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf66, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf68 = buf67[0]
        del buf67
        buf72 = reinterpret_tensor(buf65, (1576, 256), (256, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf68, (1576, 256), (256, 1), 0), reinterpret_tensor(arg37_1, (256, 256), (1, 256), 0), out=buf72)
        del arg37_1
        buf73 = reinterpret_tensor(buf72, (8, 197, 256), (50432, 256, 1), 0); del buf72  # reuse
        buf77 = reinterpret_tensor(buf68, (8, 197, 256), (50432, 256, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_193, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf73, buf54, buf61, arg32_1, arg38_1, arg39_1, arg40_1, buf77, 1576, 256, grid=grid(1576), stream=stream0)
        del arg32_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf78 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf77, (1576, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 768), (1, 256), 0), out=buf78)
        del arg41_1
        buf79 = reinterpret_tensor(buf78, (8, 197, 768), (151296, 768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf79, arg42_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg42_1
        buf80 = reinterpret_tensor(buf77, (1576, 256), (256, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf79, (1576, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 256), (1, 768), 0), out=buf80)
        del arg43_1
        buf84 = reinterpret_tensor(buf61, (8, 197, 256), (50432, 256, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_199, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf73, buf80, arg44_1, arg45_1, arg46_1, buf84, 1576, 256, grid=grid(1576), stream=stream0)
        del arg45_1
        del arg46_1
        buf85 = reinterpret_tensor(buf79, (1576, 768), (768, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg48_1, reinterpret_tensor(buf84, (1576, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf85)
        del arg47_1
        del arg48_1
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf86 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf85, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf85, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf85, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf87 = buf86[0]
        del buf86
        buf91 = reinterpret_tensor(buf84, (1576, 256), (256, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf87, (1576, 256), (256, 1), 0), reinterpret_tensor(arg49_1, (256, 256), (1, 256), 0), out=buf91)
        del arg49_1
        buf92 = reinterpret_tensor(buf91, (8, 197, 256), (50432, 256, 1), 0); del buf91  # reuse
        buf103 = reinterpret_tensor(buf87, (8, 197, 256), (50432, 256, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_199, x_204, layer_norm_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf92, buf73, buf80, arg44_1, arg50_1, arg51_1, arg52_1, buf103, 1576, 256, grid=grid(1576), stream=stream0)
        del arg44_1
        del arg50_1
        del arg51_1
        del arg52_1
        buf97 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf96, (3208, 128), (128, 1), 0), reinterpret_tensor(arg17_1, (128, 384), (1, 128), 0), out=buf97)
        del arg17_1
        buf98 = reinterpret_tensor(buf97, (8, 401, 384), (153984, 384, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf98, arg18_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg18_1
        buf99 = reinterpret_tensor(buf96, (3208, 128), (128, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf98, (3208, 384), (384, 1), 0), reinterpret_tensor(arg19_1, (384, 128), (1, 384), 0), out=buf99)
        del arg19_1
        buf111 = empty_strided_cuda((8, 1, 128), (128, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.native_layer_norm, aten.gelu]
        triton_per_fused_gelu_native_layer_norm_11.run(buf13, buf99, arg20_1, arg57_1, arg58_1, buf111, 8, 128, grid=grid(8), stream=stream0)
        del arg57_1
        del arg58_1
        buf104 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf103, (1576, 256), (256, 1), 0), reinterpret_tensor(arg53_1, (256, 768), (1, 256), 0), out=buf104)
        del arg53_1
        buf105 = reinterpret_tensor(buf104, (8, 197, 768), (151296, 768, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf105, arg54_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg54_1
        buf106 = reinterpret_tensor(buf103, (1576, 256), (256, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 768), (768, 1), 0), reinterpret_tensor(arg55_1, (768, 256), (1, 768), 0), out=buf106)
        del arg55_1
        buf112 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf111, (8, 128), (128, 1), 0), reinterpret_tensor(arg59_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf112)
        del arg59_1
        del arg60_1
        buf117 = reinterpret_tensor(buf80, (8, 197, 256), (50432, 256, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [tmp_12, layer_norm_54], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_12.run(buf112, buf92, buf106, arg56_1, arg65_1, arg66_1, buf117, 1576, 256, grid=grid(1576), stream=stream0)
        del arg65_1
        del arg66_1
        buf118 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (8, 256), (50432, 1), 0), reinterpret_tensor(arg67_1, (256, 256), (1, 256), 0), out=buf118)
        del arg67_1
        buf120 = reinterpret_tensor(buf118, (8, 1, 256), (256, 256, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf120, arg68_1, 2048, grid=grid(2048), stream=stream0)
        del arg68_1
        buf119 = reinterpret_tensor(buf73, (1576, 256), (256, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1576, 256), (256, 1), 0), reinterpret_tensor(arg69_1, (256, 256), (1, 256), 0), out=buf119)
        del arg69_1
        buf121 = reinterpret_tensor(buf54, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf119, arg70_1, buf121, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg70_1
        del buf119
        buf122 = empty_strided_cuda((32, 1, 197), (197, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf121, (32, 64, 197), (12608, 197, 1), 0), out=buf122)
        buf126 = empty_strided_cuda((8, 4, 1, 197), (788, 197, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_19], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf122, buf126, 32, 197, grid=grid(32), stream=stream0)
        buf125 = reinterpret_tensor(buf121, (1576, 256), (256, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf117, (1576, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), out=buf125)
        del arg71_1
        buf127 = reinterpret_tensor(buf117, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf125, arg72_1, buf127, 403456, grid=grid(403456), stream=stream0)
        del arg72_1
        del buf125
        buf128 = reinterpret_tensor(buf120, (32, 1, 64), (64, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf126, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf127, (32, 197, 64), (12608, 64, 1), 0), out=buf128)
        buf129 = empty_strided_cuda((8, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf128, (8, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 256), (1, 256), 0), out=buf129)
        del arg73_1
        buf130 = reinterpret_tensor(buf129, (8, 1, 256), (256, 2048, 1), 0); del buf129  # reuse
        buf135 = reinterpret_tensor(buf128, (8, 1, 256), (256, 256, 1), 0); del buf128  # reuse
        buf160 = empty_strided_cuda((8, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40, x_214, input_43, input_41, input_44], Original ATen: [aten.native_layer_norm, aten.add, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_17.run(buf130, buf92, buf106, arg56_1, buf112, arg74_1, arg61_1, arg62_1, arg75_1, arg76_1, buf135, buf160, 8, 256, grid=grid(8), stream=stream0)
        del arg61_1
        del arg62_1
        del arg74_1
        del arg75_1
        del arg76_1
        buf136 = reinterpret_tensor(buf111, (8, 128), (128, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg64_1, reinterpret_tensor(buf135, (8, 256), (256, 1), 0), reinterpret_tensor(arg63_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf136)
        del arg63_1
        del arg64_1
        buf161 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg78_1, reinterpret_tensor(buf160, (8, 256), (256, 1), 0), reinterpret_tensor(arg77_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf161)
        del arg77_1
        del arg78_1
        buf141 = empty_strided_cuda((8, 401, 128), (51328, 128, 1), torch.float32)
        buf166 = empty_strided_cuda((8, 401, 128), (51328, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tmp_14, layer_norm_56, tmp_13, layer_norm_58], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_18.run(buf136, buf13, buf99, arg20_1, buf161, arg79_1, arg93_1, arg80_1, arg94_1, buf141, buf166, 3208, 128, grid=grid(3208), stream=stream0)
        del arg79_1
        del arg80_1
        del arg93_1
        del arg94_1
        buf142 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (8, 128), (51328, 1), 0), reinterpret_tensor(arg81_1, (128, 128), (1, 128), 0), out=buf142)
        del arg81_1
        buf143 = empty_strided_cuda((3208, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (3208, 128), (128, 1), 0), reinterpret_tensor(arg83_1, (128, 128), (1, 128), 0), out=buf143)
        del arg83_1
        buf144 = reinterpret_tensor(buf142, (8, 1, 128), (128, 128, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf144, arg82_1, 1024, grid=grid(1024), stream=stream0)
        del arg82_1
        buf145 = empty_strided_cuda((8, 4, 32, 401), (51328, 12832, 401, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf143, arg84_1, buf145, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg84_1
        del buf143
        buf146 = empty_strided_cuda((32, 1, 401), (401, 401, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf145, (32, 32, 401), (12832, 401, 1), 0), out=buf146)
        buf150 = empty_strided_cuda((8, 4, 1, 401), (1632, 401, 401, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_22], Original ATen: [aten._softmax]
        triton_per_fused__softmax_21.run(buf146, buf150, 32, 401, grid=grid(32), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (3208, 128), (128, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf141, (3208, 128), (128, 1), 0), reinterpret_tensor(arg85_1, (128, 128), (1, 128), 0), out=buf149)
        del arg85_1
        buf151 = reinterpret_tensor(buf146, (32, 1, 401), (401, 12832, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_22.run(buf150, buf151, 12832, grid=grid(12832), stream=stream0)
        buf152 = reinterpret_tensor(buf141, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf149, arg86_1, buf152, 410624, grid=grid(410624), stream=stream0)
        del arg86_1
        buf153 = reinterpret_tensor(buf144, (32, 1, 32), (32, 32, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf151, reinterpret_tensor(buf152, (32, 401, 32), (12832, 32, 1), 0), out=buf153)
        buf154 = empty_strided_cuda((8, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf153, (8, 128), (128, 1), 0), reinterpret_tensor(arg87_1, (128, 128), (1, 128), 0), out=buf154)
        del arg87_1
        buf155 = reinterpret_tensor(buf154, (8, 1, 128), (128, 1024, 1), 0); del buf154  # reuse
        buf179 = reinterpret_tensor(buf153, (8, 1, 128), (128, 128, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_218, input_46, input_47], Original ATen: [aten.add, aten.native_layer_norm, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_24.run(buf155, buf136, buf13, buf99, arg20_1, arg88_1, arg89_1, arg90_1, buf179, 8, 128, grid=grid(8), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        buf167 = reinterpret_tensor(buf98, (3208, 384), (384, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg96_1, reinterpret_tensor(buf166, (3208, 128), (128, 1), 0), reinterpret_tensor(arg95_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf167)
        del arg95_1
        del arg96_1
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf167, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf167, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf167, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf169 = buf168[0]
        del buf168
        buf173 = reinterpret_tensor(buf166, (3208, 128), (128, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf169, (3208, 128), (128, 1), 0), reinterpret_tensor(arg97_1, (128, 128), (1, 128), 0), out=buf173)
        del arg97_1
        buf174 = reinterpret_tensor(buf173, (8, 401, 128), (51328, 128, 1), 0); del buf173  # reuse
        buf235 = reinterpret_tensor(buf169, (8, 401, 128), (51328, 128, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [tmp_13, x_223, layer_norm_59], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_25.run(buf174, buf161, buf13, buf99, arg20_1, arg98_1, arg99_1, arg100_1, buf235, 3208, 128, grid=grid(3208), stream=stream0)
        del arg100_1
        del arg20_1
        del arg98_1
        del arg99_1
        buf180 = reinterpret_tensor(buf160, (8, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg92_1, reinterpret_tensor(buf179, (8, 128), (128, 1), 0), reinterpret_tensor(arg91_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf180)
        del arg91_1
        del arg92_1
        buf185 = reinterpret_tensor(buf127, (8, 197, 256), (50432, 256, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [tmp_15, layer_norm_60], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_12.run(buf180, buf92, buf106, arg56_1, arg105_1, arg106_1, buf185, 1576, 256, grid=grid(1576), stream=stream0)
        del arg105_1
        del arg106_1
        buf186 = reinterpret_tensor(buf105, (1576, 768), (768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg108_1, reinterpret_tensor(buf185, (1576, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf186)
        del arg107_1
        del arg108_1
        # Topologically Sorted Source Nodes: [x_230], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf187 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf186, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf186, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf186, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf188 = buf187[0]
        del buf187
        buf192 = reinterpret_tensor(buf185, (1576, 256), (256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf188, (1576, 256), (256, 1), 0), reinterpret_tensor(arg109_1, (256, 256), (1, 256), 0), out=buf192)
        del arg109_1
        buf193 = reinterpret_tensor(buf192, (8, 197, 256), (50432, 256, 1), 0); del buf192  # reuse
        buf197 = reinterpret_tensor(buf188, (8, 197, 256), (50432, 256, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [tmp_15, x_234, layer_norm_61], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_26.run(buf193, buf180, buf92, buf106, arg56_1, arg110_1, arg111_1, arg112_1, buf197, 1576, 256, grid=grid(1576), stream=stream0)
        del arg110_1
        del arg111_1
        del arg112_1
        del arg56_1
        del buf106
        buf198 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf197, (1576, 256), (256, 1), 0), reinterpret_tensor(arg113_1, (256, 768), (1, 256), 0), out=buf198)
        del arg113_1
        buf199 = reinterpret_tensor(buf198, (8, 197, 768), (151296, 768, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf199, arg114_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg114_1
        buf200 = reinterpret_tensor(buf197, (1576, 256), (256, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf199, (1576, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 256), (1, 768), 0), out=buf200)
        del arg115_1
        buf204 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_240, layer_norm_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf193, buf200, arg116_1, arg117_1, arg118_1, buf204, 1576, 256, grid=grid(1576), stream=stream0)
        del arg117_1
        del arg118_1
        buf205 = reinterpret_tensor(buf199, (1576, 768), (768, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf204, (1576, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf205)
        del arg119_1
        del arg120_1
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf206 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf205, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf205, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf207 = buf206[0]
        del buf206
        buf211 = reinterpret_tensor(buf204, (1576, 256), (256, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf207, (1576, 256), (256, 1), 0), reinterpret_tensor(arg121_1, (256, 256), (1, 256), 0), out=buf211)
        del arg121_1
        buf212 = reinterpret_tensor(buf211, (8, 197, 256), (50432, 256, 1), 0); del buf211  # reuse
        buf216 = reinterpret_tensor(buf207, (8, 197, 256), (50432, 256, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_240, x_245, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf212, buf193, buf200, arg116_1, arg122_1, arg123_1, arg124_1, buf216, 1576, 256, grid=grid(1576), stream=stream0)
        del arg116_1
        del arg122_1
        del arg123_1
        del arg124_1
        buf217 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf216, (1576, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 768), (1, 256), 0), out=buf217)
        del arg125_1
        buf218 = reinterpret_tensor(buf217, (8, 197, 768), (151296, 768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf218, arg126_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg126_1
        buf219 = reinterpret_tensor(buf216, (1576, 256), (256, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf218, (1576, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 256), (1, 768), 0), out=buf219)
        del arg127_1
        buf223 = reinterpret_tensor(buf200, (8, 197, 256), (50432, 256, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_251, layer_norm_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf212, buf219, arg128_1, arg129_1, arg130_1, buf223, 1576, 256, grid=grid(1576), stream=stream0)
        del arg129_1
        del arg130_1
        buf224 = reinterpret_tensor(buf218, (1576, 768), (768, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg132_1, reinterpret_tensor(buf223, (1576, 256), (256, 1), 0), reinterpret_tensor(arg131_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf224)
        del arg131_1
        del arg132_1
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf225 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf224, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf224, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf224, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf226 = buf225[0]
        del buf225
        buf230 = reinterpret_tensor(buf223, (1576, 256), (256, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf226, (1576, 256), (256, 1), 0), reinterpret_tensor(arg133_1, (256, 256), (1, 256), 0), out=buf230)
        del arg133_1
        buf231 = reinterpret_tensor(buf230, (8, 197, 256), (50432, 256, 1), 0); del buf230  # reuse
        buf242 = reinterpret_tensor(buf226, (8, 197, 256), (50432, 256, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_251, x_256, layer_norm_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf231, buf212, buf219, arg128_1, arg134_1, arg135_1, arg136_1, buf242, 1576, 256, grid=grid(1576), stream=stream0)
        del arg128_1
        del arg134_1
        del arg135_1
        del arg136_1
        buf236 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf235, (3208, 128), (128, 1), 0), reinterpret_tensor(arg101_1, (128, 384), (1, 128), 0), out=buf236)
        del arg101_1
        buf237 = reinterpret_tensor(buf236, (8, 401, 384), (153984, 384, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf237, arg102_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg102_1
        buf238 = reinterpret_tensor(buf235, (3208, 128), (128, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf237, (3208, 384), (384, 1), 0), reinterpret_tensor(arg103_1, (384, 128), (1, 384), 0), out=buf238)
        del arg103_1
        buf250 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.native_layer_norm, aten.gelu]
        triton_per_fused_gelu_native_layer_norm_11.run(buf174, buf238, arg104_1, arg141_1, arg142_1, buf250, 8, 128, grid=grid(8), stream=stream0)
        del arg141_1
        del arg142_1
        buf243 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1576, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 768), (1, 256), 0), out=buf243)
        del arg137_1
        buf244 = reinterpret_tensor(buf243, (8, 197, 768), (151296, 768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf244, arg138_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg138_1
        buf245 = reinterpret_tensor(buf242, (1576, 256), (256, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf244, (1576, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 256), (1, 768), 0), out=buf245)
        del arg139_1
        buf251 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg144_1, reinterpret_tensor(buf250, (8, 128), (128, 1), 0), reinterpret_tensor(arg143_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf251)
        del arg143_1
        del arg144_1
        buf256 = reinterpret_tensor(buf219, (8, 197, 256), (50432, 256, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [tmp_16, layer_norm_68], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_12.run(buf251, buf231, buf245, arg140_1, arg149_1, arg150_1, buf256, 1576, 256, grid=grid(1576), stream=stream0)
        del arg149_1
        del arg150_1
        buf257 = reinterpret_tensor(buf135, (8, 256), (256, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [linear_132], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (8, 256), (50432, 1), 0), reinterpret_tensor(arg151_1, (256, 256), (1, 256), 0), out=buf257)
        del arg151_1
        buf259 = reinterpret_tensor(buf257, (8, 1, 256), (256, 256, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [linear_132], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf259, arg152_1, 2048, grid=grid(2048), stream=stream0)
        del arg152_1
        buf258 = reinterpret_tensor(buf212, (1576, 256), (256, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1576, 256), (256, 1), 0), reinterpret_tensor(arg153_1, (256, 256), (1, 256), 0), out=buf258)
        del arg153_1
        buf260 = reinterpret_tensor(buf193, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf258, arg154_1, buf260, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg154_1
        del buf258
        buf261 = reinterpret_tensor(buf126, (32, 1, 197), (197, 197, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf259, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf260, (32, 64, 197), (12608, 197, 1), 0), out=buf261)
        buf265 = reinterpret_tensor(buf122, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [attn_25], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf261, buf265, 32, 197, grid=grid(32), stream=stream0)
        buf264 = reinterpret_tensor(buf260, (1576, 256), (256, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf256, (1576, 256), (256, 1), 0), reinterpret_tensor(arg155_1, (256, 256), (1, 256), 0), out=buf264)
        del arg155_1
        buf266 = reinterpret_tensor(buf256, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf264, arg156_1, buf266, 403456, grid=grid(403456), stream=stream0)
        del arg156_1
        del buf264
        buf267 = reinterpret_tensor(buf259, (32, 1, 64), (64, 64, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf265, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf266, (32, 197, 64), (12608, 64, 1), 0), out=buf267)
        buf268 = reinterpret_tensor(buf130, (8, 256), (256, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf267, (8, 256), (256, 1), 0), reinterpret_tensor(arg157_1, (256, 256), (1, 256), 0), out=buf268)
        del arg157_1
        buf269 = reinterpret_tensor(buf268, (8, 1, 256), (256, 2048, 1), 0); del buf268  # reuse
        buf274 = reinterpret_tensor(buf267, (8, 1, 256), (256, 256, 1), 0); del buf267  # reuse
        buf299 = reinterpret_tensor(buf112, (8, 1, 256), (256, 256, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [input_52, x_266, input_55, input_53, input_56], Original ATen: [aten.native_layer_norm, aten.add, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_17.run(buf269, buf231, buf245, arg140_1, buf251, arg158_1, arg145_1, arg146_1, arg159_1, arg160_1, buf274, buf299, 8, 256, grid=grid(8), stream=stream0)
        del arg145_1
        del arg146_1
        del arg158_1
        del arg159_1
        del arg160_1
        buf275 = reinterpret_tensor(buf250, (8, 128), (128, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg148_1, reinterpret_tensor(buf274, (8, 256), (256, 1), 0), reinterpret_tensor(arg147_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf275)
        del arg147_1
        del arg148_1
        buf300 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg162_1, reinterpret_tensor(buf299, (8, 256), (256, 1), 0), reinterpret_tensor(arg161_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf300)
        del arg161_1
        del arg162_1
        buf280 = reinterpret_tensor(buf99, (8, 401, 128), (51328, 128, 1), 0); del buf99  # reuse
        buf305 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [tmp_18, layer_norm_70, tmp_17, layer_norm_72], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_18.run(buf275, buf174, buf238, arg104_1, buf300, arg163_1, arg177_1, arg164_1, arg178_1, buf280, buf305, 3208, 128, grid=grid(3208), stream=stream0)
        del arg163_1
        del arg164_1
        del arg177_1
        del arg178_1
        buf281 = reinterpret_tensor(buf155, (8, 128), (128, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (8, 128), (51328, 1), 0), reinterpret_tensor(arg165_1, (128, 128), (1, 128), 0), out=buf281)
        del arg165_1
        buf282 = reinterpret_tensor(buf152, (3208, 128), (128, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (3208, 128), (128, 1), 0), reinterpret_tensor(arg167_1, (128, 128), (1, 128), 0), out=buf282)
        del arg167_1
        buf283 = reinterpret_tensor(buf281, (8, 1, 128), (128, 128, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf283, arg166_1, 1024, grid=grid(1024), stream=stream0)
        del arg166_1
        buf284 = reinterpret_tensor(buf149, (8, 4, 32, 401), (51328, 12832, 401, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf282, arg168_1, buf284, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg168_1
        del buf282
        buf285 = reinterpret_tensor(buf151, (32, 1, 401), (401, 401, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf284, (32, 32, 401), (12832, 401, 1), 0), out=buf285)
        buf289 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [attn_28], Original ATen: [aten._softmax]
        triton_per_fused__softmax_21.run(buf285, buf289, 32, 401, grid=grid(32), stream=stream0)
        buf288 = reinterpret_tensor(buf284, (3208, 128), (128, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (3208, 128), (128, 1), 0), reinterpret_tensor(arg169_1, (128, 128), (1, 128), 0), out=buf288)
        del arg169_1
        buf290 = reinterpret_tensor(buf285, (32, 1, 401), (401, 12832, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_22.run(buf289, buf290, 12832, grid=grid(12832), stream=stream0)
        buf291 = reinterpret_tensor(buf280, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf288, arg170_1, buf291, 410624, grid=grid(410624), stream=stream0)
        del arg170_1
        buf292 = reinterpret_tensor(buf283, (32, 1, 32), (32, 32, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf290, reinterpret_tensor(buf291, (32, 401, 32), (12832, 32, 1), 0), out=buf292)
        buf293 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf292, (8, 128), (128, 1), 0), reinterpret_tensor(arg171_1, (128, 128), (1, 128), 0), out=buf293)
        del arg171_1
        buf294 = reinterpret_tensor(buf293, (8, 1, 128), (128, 1024, 1), 0); del buf293  # reuse
        buf318 = reinterpret_tensor(buf292, (8, 1, 128), (128, 128, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_270, input_58, input_59], Original ATen: [aten.add, aten.native_layer_norm, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_24.run(buf294, buf275, buf174, buf238, arg104_1, arg172_1, arg173_1, arg174_1, buf318, 8, 128, grid=grid(8), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del buf275
        buf306 = reinterpret_tensor(buf237, (3208, 384), (384, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg180_1, reinterpret_tensor(buf305, (3208, 128), (128, 1), 0), reinterpret_tensor(arg179_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf306)
        del arg179_1
        del arg180_1
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf307 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf306, (8, 4, 401, 32), (153984, 32, 384, 1), 0), reinterpret_tensor(buf306, (8, 4, 401, 32), (153984, 32, 384, 1), 128), reinterpret_tensor(buf306, (8, 4, 401, 32), (153984, 32, 384, 1), 256), None, False)
        buf308 = buf307[0]
        del buf307
        buf312 = reinterpret_tensor(buf305, (3208, 128), (128, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf308, (3208, 128), (128, 1), 0), reinterpret_tensor(arg181_1, (128, 128), (1, 128), 0), out=buf312)
        del arg181_1
        buf313 = reinterpret_tensor(buf312, (8, 401, 128), (51328, 128, 1), 0); del buf312  # reuse
        buf374 = reinterpret_tensor(buf308, (8, 401, 128), (51328, 128, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [tmp_17, x_275, layer_norm_73], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_25.run(buf313, buf300, buf174, buf238, arg104_1, arg182_1, arg183_1, arg184_1, buf374, 3208, 128, grid=grid(3208), stream=stream0)
        del arg104_1
        del arg182_1
        del arg183_1
        del arg184_1
        buf319 = reinterpret_tensor(buf299, (8, 256), (256, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg176_1, reinterpret_tensor(buf318, (8, 128), (128, 1), 0), reinterpret_tensor(arg175_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf319)
        del arg175_1
        del arg176_1
        buf324 = reinterpret_tensor(buf266, (8, 197, 256), (50432, 256, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [tmp_19, layer_norm_74], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_12.run(buf319, buf231, buf245, arg140_1, arg189_1, arg190_1, buf324, 1576, 256, grid=grid(1576), stream=stream0)
        del arg189_1
        del arg190_1
        buf325 = reinterpret_tensor(buf244, (1576, 768), (768, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg192_1, reinterpret_tensor(buf324, (1576, 256), (256, 1), 0), reinterpret_tensor(arg191_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf325)
        del arg191_1
        del arg192_1
        # Topologically Sorted Source Nodes: [x_282], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf326 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf325, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf325, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf325, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf327 = buf326[0]
        del buf326
        buf331 = reinterpret_tensor(buf324, (1576, 256), (256, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf327, (1576, 256), (256, 1), 0), reinterpret_tensor(arg193_1, (256, 256), (1, 256), 0), out=buf331)
        del arg193_1
        buf332 = reinterpret_tensor(buf331, (8, 197, 256), (50432, 256, 1), 0); del buf331  # reuse
        buf336 = reinterpret_tensor(buf327, (8, 197, 256), (50432, 256, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [tmp_19, x_286, layer_norm_75], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        triton_per_fused_add_cat_native_layer_norm_26.run(buf332, buf319, buf231, buf245, arg140_1, arg194_1, arg195_1, arg196_1, buf336, 1576, 256, grid=grid(1576), stream=stream0)
        del arg140_1
        del arg194_1
        del arg195_1
        del arg196_1
        del buf231
        buf337 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (1576, 256), (256, 1), 0), reinterpret_tensor(arg197_1, (256, 768), (1, 256), 0), out=buf337)
        del arg197_1
        buf338 = reinterpret_tensor(buf337, (8, 197, 768), (151296, 768, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_288], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf338, arg198_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg198_1
        buf339 = reinterpret_tensor(buf336, (1576, 256), (256, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf338, (1576, 768), (768, 1), 0), reinterpret_tensor(arg199_1, (768, 256), (1, 768), 0), out=buf339)
        del arg199_1
        buf343 = reinterpret_tensor(buf245, (8, 197, 256), (50432, 256, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_292, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf332, buf339, arg200_1, arg201_1, arg202_1, buf343, 1576, 256, grid=grid(1576), stream=stream0)
        del arg201_1
        del arg202_1
        buf344 = reinterpret_tensor(buf338, (1576, 768), (768, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [linear_150], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg204_1, reinterpret_tensor(buf343, (1576, 256), (256, 1), 0), reinterpret_tensor(arg203_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf344)
        del arg203_1
        del arg204_1
        # Topologically Sorted Source Nodes: [x_293], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf345 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf344, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf344, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf344, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf346 = buf345[0]
        del buf345
        buf350 = reinterpret_tensor(buf343, (1576, 256), (256, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (1576, 256), (256, 1), 0), reinterpret_tensor(arg205_1, (256, 256), (1, 256), 0), out=buf350)
        del arg205_1
        buf351 = reinterpret_tensor(buf350, (8, 197, 256), (50432, 256, 1), 0); del buf350  # reuse
        buf355 = reinterpret_tensor(buf346, (8, 197, 256), (50432, 256, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_292, x_297, layer_norm_77], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf351, buf332, buf339, arg200_1, arg206_1, arg207_1, arg208_1, buf355, 1576, 256, grid=grid(1576), stream=stream0)
        del arg200_1
        del arg206_1
        del arg207_1
        del arg208_1
        buf356 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (1576, 256), (256, 1), 0), reinterpret_tensor(arg209_1, (256, 768), (1, 256), 0), out=buf356)
        del arg209_1
        buf357 = reinterpret_tensor(buf356, (8, 197, 768), (151296, 768, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf357, arg210_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg210_1
        buf358 = reinterpret_tensor(buf355, (1576, 256), (256, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf357, (1576, 768), (768, 1), 0), reinterpret_tensor(arg211_1, (768, 256), (1, 768), 0), out=buf358)
        del arg211_1
        buf362 = reinterpret_tensor(buf339, (8, 197, 256), (50432, 256, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [x_303, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_8.run(buf351, buf358, arg212_1, arg213_1, arg214_1, buf362, 1576, 256, grid=grid(1576), stream=stream0)
        del arg213_1
        del arg214_1
        buf363 = reinterpret_tensor(buf357, (1576, 768), (768, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg216_1, reinterpret_tensor(buf362, (1576, 256), (256, 1), 0), reinterpret_tensor(arg215_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf363)
        del arg215_1
        del arg216_1
        # Topologically Sorted Source Nodes: [x_304], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf364 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf363, (8, 4, 197, 64), (151296, 64, 768, 1), 0), reinterpret_tensor(buf363, (8, 4, 197, 64), (151296, 64, 768, 1), 256), reinterpret_tensor(buf363, (8, 4, 197, 64), (151296, 64, 768, 1), 512), None, False)
        buf365 = buf364[0]
        del buf364
        buf369 = reinterpret_tensor(buf362, (1576, 256), (256, 1), 0); del buf362  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf365, (1576, 256), (256, 1), 0), reinterpret_tensor(arg217_1, (256, 256), (1, 256), 0), out=buf369)
        del arg217_1
        buf370 = reinterpret_tensor(buf369, (8, 197, 256), (50432, 256, 1), 0); del buf369  # reuse
        buf381 = reinterpret_tensor(buf365, (8, 197, 256), (50432, 256, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_303, x_308, layer_norm_79], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_9.run(buf370, buf351, buf358, arg212_1, arg218_1, arg219_1, arg220_1, buf381, 1576, 256, grid=grid(1576), stream=stream0)
        del arg212_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf375 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf374, (3208, 128), (128, 1), 0), reinterpret_tensor(arg185_1, (128, 384), (1, 128), 0), out=buf375)
        del arg185_1
        buf376 = reinterpret_tensor(buf375, (8, 401, 384), (153984, 384, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_10.run(buf376, arg186_1, 1231872, grid=grid(1231872), stream=stream0)
        del arg186_1
        buf377 = reinterpret_tensor(buf374, (3208, 128), (128, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf376, (3208, 384), (384, 1), 0), reinterpret_tensor(arg187_1, (384, 128), (1, 384), 0), out=buf377)
        del arg187_1
        del buf376
        buf389 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten.native_layer_norm, aten.gelu]
        triton_per_fused_gelu_native_layer_norm_11.run(buf313, buf377, arg188_1, arg225_1, arg226_1, buf389, 8, 128, grid=grid(8), stream=stream0)
        del arg225_1
        del arg226_1
        buf382 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf381, (1576, 256), (256, 1), 0), reinterpret_tensor(arg221_1, (256, 768), (1, 256), 0), out=buf382)
        del arg221_1
        buf383 = reinterpret_tensor(buf382, (8, 197, 768), (151296, 768, 1), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [x_310], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_7.run(buf383, arg222_1, 1210368, grid=grid(1210368), stream=stream0)
        del arg222_1
        buf384 = reinterpret_tensor(buf381, (1576, 256), (256, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (1576, 768), (768, 1), 0), reinterpret_tensor(arg223_1, (768, 256), (1, 768), 0), out=buf384)
        del arg223_1
        del buf383
        buf390 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg228_1, reinterpret_tensor(buf389, (8, 128), (128, 1), 0), reinterpret_tensor(arg227_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf390)
        del arg227_1
        del arg228_1
        buf395 = reinterpret_tensor(buf358, (8, 197, 256), (50432, 256, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [tmp_20, layer_norm_82], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_12.run(buf390, buf370, buf384, arg224_1, arg233_1, arg234_1, buf395, 1576, 256, grid=grid(1576), stream=stream0)
        del arg233_1
        del arg234_1
        buf396 = reinterpret_tensor(buf274, (8, 256), (256, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (8, 256), (50432, 1), 0), reinterpret_tensor(arg235_1, (256, 256), (1, 256), 0), out=buf396)
        del arg235_1
        buf398 = reinterpret_tensor(buf396, (8, 1, 256), (256, 256, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf398, arg236_1, 2048, grid=grid(2048), stream=stream0)
        del arg236_1
        buf397 = reinterpret_tensor(buf351, (1576, 256), (256, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf395, (1576, 256), (256, 1), 0), reinterpret_tensor(arg237_1, (256, 256), (1, 256), 0), out=buf397)
        del arg237_1
        buf399 = reinterpret_tensor(buf332, (8, 4, 64, 197), (50432, 12608, 197, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf397, arg238_1, buf399, 2048, 197, grid=grid(2048, 197), stream=stream0)
        del arg238_1
        del buf397
        buf400 = reinterpret_tensor(buf265, (32, 1, 197), (197, 197, 1), 0); del buf265  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf398, (32, 1, 64), (64, 0, 1), 0), reinterpret_tensor(buf399, (32, 64, 197), (12608, 197, 1), 0), out=buf400)
        buf404 = reinterpret_tensor(buf261, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [attn_31], Original ATen: [aten._softmax]
        triton_per_fused__softmax_15.run(buf400, buf404, 32, 197, grid=grid(32), stream=stream0)
        del buf400
        buf403 = reinterpret_tensor(buf399, (1576, 256), (256, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf395, (1576, 256), (256, 1), 0), reinterpret_tensor(arg239_1, (256, 256), (1, 256), 0), out=buf403)
        del arg239_1
        buf405 = reinterpret_tensor(buf395, (8, 4, 197, 64), (50432, 12608, 64, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf403, arg240_1, buf405, 403456, grid=grid(403456), stream=stream0)
        del arg240_1
        del buf403
        buf406 = reinterpret_tensor(buf398, (32, 1, 64), (64, 64, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf404, (32, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf405, (32, 197, 64), (12608, 64, 1), 0), out=buf406)
        del buf404
        buf407 = reinterpret_tensor(buf269, (8, 256), (256, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf406, (8, 256), (256, 1), 0), reinterpret_tensor(arg241_1, (256, 256), (1, 256), 0), out=buf407)
        del arg241_1
        buf408 = reinterpret_tensor(buf407, (8, 1, 256), (256, 2048, 1), 0); del buf407  # reuse
        buf413 = reinterpret_tensor(buf406, (8, 1, 256), (256, 256, 1), 0); del buf406  # reuse
        buf438 = reinterpret_tensor(buf251, (8, 1, 256), (256, 256, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [input_64, x_318, input_67, input_65, input_68], Original ATen: [aten.native_layer_norm, aten.add, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_17.run(buf408, buf370, buf384, arg224_1, buf390, arg242_1, arg229_1, arg230_1, arg243_1, arg244_1, buf413, buf438, 8, 256, grid=grid(8), stream=stream0)
        del arg229_1
        del arg230_1
        del arg242_1
        del arg243_1
        del arg244_1
        del buf390
        del buf408
        buf414 = reinterpret_tensor(buf389, (8, 128), (128, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg232_1, reinterpret_tensor(buf413, (8, 256), (256, 1), 0), reinterpret_tensor(arg231_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf414)
        del arg231_1
        del arg232_1
        del buf413
        buf439 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg246_1, reinterpret_tensor(buf438, (8, 256), (256, 1), 0), reinterpret_tensor(arg245_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf439)
        del arg245_1
        del arg246_1
        buf449 = reinterpret_tensor(buf238, (8, 401, 128), (51328, 128, 1), 0); del buf238  # reuse
        buf419 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [tmp_22, layer_norm_84, tmp_21, x_323], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_27.run(buf414, buf313, buf377, arg188_1, buf439, arg247_1, arg261_1, arg248_1, buf449, buf419, 3208, 128, grid=grid(3208), stream=stream0)
        del arg247_1
        del arg248_1
        del arg261_1
        buf420 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [linear_165], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (8, 128), (51328, 1), 0), reinterpret_tensor(arg249_1, (128, 128), (1, 128), 0), out=buf420)
        del arg249_1
        buf421 = reinterpret_tensor(buf291, (3208, 128), (128, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (3208, 128), (128, 1), 0), reinterpret_tensor(arg251_1, (128, 128), (1, 128), 0), out=buf421)
        del arg251_1
        buf422 = reinterpret_tensor(buf420, (8, 1, 128), (128, 128, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [linear_165], Original ATen: [aten.add]
        triton_poi_fused_add_19.run(buf422, arg250_1, 1024, grid=grid(1024), stream=stream0)
        del arg250_1
        buf423 = reinterpret_tensor(buf288, (8, 4, 32, 401), (51328, 12832, 401, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf421, arg252_1, buf423, 1024, 401, grid=grid(1024, 401), stream=stream0)
        del arg252_1
        del buf421
        buf424 = reinterpret_tensor(buf290, (32, 1, 401), (401, 401, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (32, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf423, (32, 32, 401), (12832, 401, 1), 0), out=buf424)
        buf428 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [attn_34], Original ATen: [aten._softmax]
        triton_per_fused__softmax_21.run(buf424, buf428, 32, 401, grid=grid(32), stream=stream0)
        buf427 = reinterpret_tensor(buf423, (3208, 128), (128, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (3208, 128), (128, 1), 0), reinterpret_tensor(arg253_1, (128, 128), (1, 128), 0), out=buf427)
        del arg253_1
        buf429 = reinterpret_tensor(buf424, (32, 1, 401), (401, 12832, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_22.run(buf428, buf429, 12832, grid=grid(12832), stream=stream0)
        del buf428
        buf430 = reinterpret_tensor(buf419, (8, 4, 401, 32), (51328, 12832, 32, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf427, arg254_1, buf430, 410624, grid=grid(410624), stream=stream0)
        del arg254_1
        del buf427
        buf431 = reinterpret_tensor(buf422, (32, 1, 32), (32, 32, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf429, reinterpret_tensor(buf430, (32, 401, 32), (12832, 32, 1), 0), out=buf431)
        del buf429
        del buf430
        buf432 = reinterpret_tensor(buf294, (8, 128), (128, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf431, (8, 128), (128, 1), 0), reinterpret_tensor(arg255_1, (128, 128), (1, 128), 0), out=buf432)
        del arg255_1
        buf433 = reinterpret_tensor(buf432, (8, 1, 128), (128, 1024, 1), 0); del buf432  # reuse
        buf444 = reinterpret_tensor(buf431, (8, 1, 128), (128, 128, 1), 0); del buf431  # reuse
        # Topologically Sorted Source Nodes: [x_322, input_70, input_71], Original ATen: [aten.add, aten.native_layer_norm, aten.gelu]
        triton_per_fused_add_gelu_native_layer_norm_24.run(buf433, buf414, buf313, buf377, arg188_1, arg256_1, arg257_1, arg258_1, buf444, 8, 128, grid=grid(8), stream=stream0)
        del arg188_1
        del arg256_1
        del arg257_1
        del arg258_1
        del buf313
        del buf377
        del buf414
        del buf433
        buf445 = reinterpret_tensor(buf438, (8, 256), (256, 1), 0); del buf438  # reuse
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg260_1, reinterpret_tensor(buf444, (8, 128), (128, 1), 0), reinterpret_tensor(arg259_1, (128, 256), (1, 128), 0), alpha=1, beta=1, out=buf445)
        del arg259_1
        del arg260_1
        buf452 = reinterpret_tensor(buf405, (8, 197, 256), (50432, 256, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [tmp_23, x_324], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_28.run(buf445, buf370, buf384, arg224_1, arg263_1, buf452, 1576, 256, grid=grid(1576), stream=stream0)
        del arg224_1
        del arg263_1
        del buf370
        del buf384
        buf450 = reinterpret_tensor(buf444, (8, 128), (128, 1), 0); del buf444  # reuse
        # Topologically Sorted Source Nodes: [dropout_102], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf449, arg262_1, buf450, 1024, grid=grid(1024), stream=stream0)
        del arg262_1
        del buf449
        buf455 = empty_strided_cuda((16, 1000), (1000, 1), torch.float32)
        buf451 = reinterpret_tensor(buf455, (8, 1000), (1000, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [dropout_102, linear_170], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg266_1, buf450, reinterpret_tensor(arg265_1, (128, 1000), (1, 128), 0), alpha=1, beta=1, out=buf451)
        del arg265_1
        del arg266_1
        del buf450
        buf453 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [dropout_103], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf452, arg264_1, buf453, 2048, grid=grid(2048), stream=stream0)
        del arg264_1
        del buf452
        buf454 = reinterpret_tensor(buf455, (8, 1000), (1000, 1), 8000)  # alias
        # Topologically Sorted Source Nodes: [dropout_103, linear_171], Original ATen: [aten.clone, aten.addmm]
        extern_kernels.addmm(arg268_1, buf453, reinterpret_tensor(arg267_1, (256, 1000), (1, 256), 0), alpha=1, beta=1, out=buf454)
        del arg267_1
        del arg268_1
        del buf453
        buf456 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.mean]
        triton_poi_fused_mean_31.run(buf455, buf456, 8000, grid=grid(8000), stream=stream0)
        del buf451
        del buf454
        del buf455
    return (buf456, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg77_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
