
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 19456
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2432
    x2 = xindex
    y1 = (yindex // 2432)
    y3 = yindex
    tmp43 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1, 1], 2432, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp1
    tmp21 = tl.full([1, 1], 1344, tl.int64)
    tmp22 = tmp19 < tmp21
    tmp23 = tmp22 & tmp16
    tmp24 = tl.full([1, 1], 1280, tl.int64)
    tmp25 = tmp19 < tmp24
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr5 + (x2 + (196*((-1024) + y0)) + (250880*y1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp19 >= tmp24
    tmp29 = tmp28 & tmp23
    tmp30 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-1280) + ((-1024) + y0))), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp25, tmp27, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp19 >= tmp21
    tmp35 = tl.full([1, 1], 1408, tl.int64)
    tmp36 = tmp19 < tmp35
    tmp37 = tmp34 & tmp16
    tmp38 = tl.load(in_ptr4 + (1024 + (1088*x2) + (213248*y1) + ((-1344) + ((-1024) + y0))), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp22, tmp33, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp16, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp15, tmp41)
    tmp44 = tmp42 - tmp43
    tmp46 = 0.001
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.sqrt(tmp47)
    tmp49 = tl.full([1, 1], 1, tl.int32)
    tmp50 = tmp49 / tmp48
    tmp51 = 1.0
    tmp52 = tmp50 * tmp51
    tmp53 = tmp44 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = tl.full([1, 1], 0, tl.int32)
    tmp59 = triton_helpers.maximum(tmp58, tmp57)
    tmp61 = tmp42 - tmp60
    tmp63 = tmp62 + tmp46
    tmp64 = libdevice.sqrt(tmp63)
    tmp65 = tmp49 / tmp64
    tmp66 = tmp65 * tmp51
    tmp67 = tmp61 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(tmp58, tmp71)
    tl.store(out_ptr1 + (y0 + (2432*x2) + (476672*y1)), tmp59, xmask)
    tl.store(out_ptr2 + (y0 + (2432*x2) + (476672*y1)), tmp72, xmask)
