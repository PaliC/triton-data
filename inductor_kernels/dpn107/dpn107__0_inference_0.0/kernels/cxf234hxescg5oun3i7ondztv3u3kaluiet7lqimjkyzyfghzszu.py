
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8704
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1088
    x2 = xindex
    y1 = (yindex // 1088)
    y3 = yindex
    tmp31 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + ((576*x2) + (451584*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 1088, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tmp19 & tmp14
    tmp21 = tl.load(in_ptr4 + (x2 + (784*((-512) + y0)) + (401408*y1)), tmp20 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp17 >= tmp3
    tmp23 = tl.full([1, 1], 576, tl.int64)
    tmp24 = tmp17 < tmp23
    tmp25 = tmp22 & tmp14
    tmp26 = tl.load(in_ptr3 + (512 + (576*x2) + (451584*y1) + ((-512) + ((-512) + y0))), tmp25 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp19, tmp21, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp14, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp13, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tl.full([1, 1], 1, tl.int32)
    tmp38 = tmp37 / tmp36
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp32 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1, 1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tl.store(out_ptr1 + (y0 + (1088*x2) + (852992*y1)), tmp47, xmask & ymask)
