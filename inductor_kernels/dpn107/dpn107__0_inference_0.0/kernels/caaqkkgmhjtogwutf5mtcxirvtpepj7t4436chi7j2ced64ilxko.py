
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15872
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1984
    x2 = xindex
    y1 = (yindex // 1984)
    y3 = yindex
    tmp28 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (376320*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((1088*x2) + (213248*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 1984, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.broadcast_to((-1024) + y0, [XBLOCK, YBLOCK])
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1, 1], 896, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + (x2 + (196*((-1024) + y0)) + (376320*y1)), tmp17 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp13 >= tmp15
    tmp20 = tl.full([1, 1], 960, tl.int64)
    tmp21 = tmp13 < tmp20
    tmp22 = tmp19 & tmp10
    tmp23 = tl.load(in_ptr1 + (1024 + (1088*x2) + (213248*y1) + ((-896) + ((-1024) + y0))), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp16, tmp18, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp9, tmp26)
    tmp29 = tmp27 - tmp28
    tmp31 = 0.001
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full([1, 1], 0, tl.int32)
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tl.store(out_ptr1 + (y0 + (1984*x2) + (388864*y1)), tmp44, xmask & ymask)
