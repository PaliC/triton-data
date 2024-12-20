
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1080
    x2 = xindex
    y1 = (yindex // 1080)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 432, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 648, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((216*x2) + (381024*y1) + ((-432) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 864, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 1080, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr0 + (x2 + (1792*y3)), tmp95, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
