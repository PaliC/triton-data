
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 270
    x2 = xindex
    y1 = (yindex // 270)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 54, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((54*x2) + (372006*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 108, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((54*x2) + (372006*y1) + ((-54) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 - tmp11
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr6 + ((54*x2) + (372006*y1) + ((-54) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp9, tmp27, tmp28)
    tmp30 = tmp0 >= tmp7
    tmp31 = tl.full([1, 1], 162, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr7 + ((54*x2) + (372006*y1) + ((-108) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp0 >= tmp31
    tmp36 = tl.full([1, 1], 216, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + ((54*x2) + (372006*y1) + ((-162) + y0)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr9 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 - tmp40
    tmp42 = tl.load(in_ptr10 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp42 + tmp14
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tmp17 / tmp44
    tmp46 = tmp45 * tmp19
    tmp47 = tmp41 * tmp46
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 * tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.load(in_ptr13 + ((54*x2) + (372006*y1) + ((-162) + y0)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp38, tmp53, tmp54)
    tmp56 = tmp0 >= tmp36
    tmp57 = tl.full([1, 1], 270, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = tl.load(in_ptr14 + ((54*x2) + (372006*y1) + ((-216) + y0)), tmp56 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.where(tmp38, tmp55, tmp59)
    tmp61 = tl.where(tmp33, tmp34, tmp60)
    tmp62 = tl.where(tmp9, tmp29, tmp61)
    tmp63 = tl.where(tmp4, tmp5, tmp62)
    tmp64 = tl.full([1, 1], 0, tl.int32)
    tmp65 = triton_helpers.maximum(tmp64, tmp63)
    tl.store(out_ptr0 + (x2 + (6912*y3)), tmp63, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (270*x2) + (1860030*y1)), tmp65, xmask & ymask)
