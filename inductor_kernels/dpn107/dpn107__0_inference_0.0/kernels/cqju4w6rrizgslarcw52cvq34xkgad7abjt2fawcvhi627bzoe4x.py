
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2048
    x2 = xindex
    y1 = (yindex // 2048)
    y3 = yindex
    tmp38 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (376320*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 2048, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1, 1], 960, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1, 1], 896, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + (x2 + (196*((-1024) + y0)) + (376320*y1)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp15 >= tmp20
    tmp25 = tmp24 & tmp19
    tmp26 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-896) + ((-1024) + y0))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp21, tmp23, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp19, tmp27, tmp28)
    tmp30 = tmp15 >= tmp17
    tmp31 = tmp15 < tmp3
    tmp32 = tmp30 & tmp12
    tmp33 = tl.load(in_ptr2 + (1024 + (1088*x2) + (213248*y1) + ((-960) + ((-1024) + y0))), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp18, tmp29, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp12, tmp34, tmp35)
    tmp37 = tl.where(tmp4, tmp11, tmp36)
    tmp39 = tmp37 - tmp38
    tmp41 = 0.001
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.sqrt(tmp42)
    tmp44 = tl.full([1, 1], 1, tl.int32)
    tmp45 = tmp44 / tmp43
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp39 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full([1, 1], 0, tl.int32)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tl.store(out_ptr1 + (y0 + (2048*x2) + (401408*y1)), tmp54, xmask)
