
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14848
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1856
    x2 = xindex
    y1 = (yindex // 1856)
    y3 = yindex
    tmp41 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((1024*x2) + (200704*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1856, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 768, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1, 1], 704, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr4 + (x2 + (196*((-1024) + y0)) + (137984*y1)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp17 >= tmp22
    tmp27 = tmp26 & tmp21
    tmp28 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-704) + ((-1024) + y0))), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp23, tmp25, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp21, tmp29, tmp30)
    tmp32 = tmp17 >= tmp19
    tmp33 = tl.full([1, 1], 832, tl.int64)
    tmp34 = tmp17 < tmp33
    tmp35 = tmp32 & tmp14
    tmp36 = tl.load(in_ptr3 + (1024 + (1088*x2) + (213248*y1) + ((-768) + ((-1024) + y0))), tmp35 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.where(tmp20, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp14, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp13, tmp39)
    tmp42 = tmp40 - tmp41
    tmp44 = 0.001
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = tl.full([1, 1], 1, tl.int32)
    tmp48 = tmp47 / tmp46
    tmp49 = 1.0
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tl.full([1, 1], 0, tl.int32)
    tmp57 = triton_helpers.maximum(tmp56, tmp55)
    tl.store(out_ptr1 + (y0 + (1856*x2) + (363776*y1)), tmp57, xmask & ymask)
