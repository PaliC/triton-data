
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 896
    x2 = xindex
    y1 = (yindex // 896)
    y3 = yindex
    tmp24 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x2) + (401408*y1) + y0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 896, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-512) + y0, [XBLOCK, YBLOCK])
    tmp10 = tmp9 >= tmp1
    tmp11 = tl.full([1, 1], 320, tl.int64)
    tmp12 = tmp9 < tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr1 + (x2 + (784*((-512) + y0)) + (250880*y1)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp11
    tmp16 = tl.full([1, 1], 384, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp6
    tmp19 = tl.load(in_ptr2 + (512 + (576*x2) + (451584*y1) + ((-320) + ((-512) + y0))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp12, tmp14, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tl.where(tmp4, tmp5, tmp22)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.001
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1, 1], 1, tl.int32)
    tmp31 = tmp30 / tmp29
    tmp32 = 1.0
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tl.full([1, 1], 0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tl.store(out_ptr1 + (y0 + (896*x2) + (702464*y1)), tmp40, xmask)
