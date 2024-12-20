
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((32*x1) + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((32*x1) + ((-96) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 - tmp16
    tmp18 = tl.load(in_ptr4 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tl.full([1], 1, tl.int32)
    tmp23 = tmp22 / tmp21
    tmp24 = 1.0
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 * tmp25
    tmp27 = tl.load(in_ptr5 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr6 + ((-96) + x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp14, tmp32, tmp33)
    tmp35 = tmp0 >= tmp12
    tmp36 = tl.full([1], 192, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr7 + ((64*x1) + ((-128) + x0)), tmp35, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp14, tmp34, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, None)
