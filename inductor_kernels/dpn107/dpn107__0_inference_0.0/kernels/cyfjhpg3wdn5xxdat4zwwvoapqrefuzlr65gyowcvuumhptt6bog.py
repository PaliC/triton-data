
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    y3 = yindex
    tmp39 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((296*x2) + (928256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((276*x2) + (865536*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 336, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 60, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 40, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (256 + (296*x2) + (928256*y1) + ((-256) + y0)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (256 + (276*x2) + (865536*y1) + ((-40) + ((-256) + y0))), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tl.full([1, 1], 80, tl.int64)
    tmp32 = tmp15 < tmp31
    tmp33 = tmp30 & tmp12
    tmp34 = tl.load(in_ptr2 + (256 + (276*x2) + (865536*y1) + ((-60) + ((-256) + y0))), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp18, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp11, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 0.001
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tl.full([1, 1], 1, tl.int32)
    tmp46 = tmp45 / tmp44
    tmp47 = 1.0
    tmp48 = tmp46 * tmp47
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1, 1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tl.store(out_ptr1 + (y0 + (336*x2) + (1053696*y1)), tmp55, xmask & ymask)
