
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp28 = tl.load(in_out_ptr0 + (x2), None)
    tmp29 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((12*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 24, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((12*x1) + ((-12) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp21 = tl.load(in_ptr4 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr5 + ((-12) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp6, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp5, tmp26)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp31 + tmp13
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tmp16 / tmp33
    tmp35 = tmp34 * tmp18
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp27 + tmp40
    tl.store(in_out_ptr0 + (x2), tmp41, None)
