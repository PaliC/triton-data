
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_cat_mean_silu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 624
    x1 = (xindex // 624)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    x3 = xindex
    tmp34 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 156, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((156*r2) + (30576*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 312, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tmp6 & tmp8
        tmp10 = tl.load(in_ptr1 + ((156*r2) + (30576*x1) + ((-156) + x0)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp0 >= tmp7
        tmp12 = tl.full([1, 1], 468, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tmp11 & tmp13
        tmp15 = tl.load(in_ptr2 + ((156*r2) + (30576*x1) + ((-312) + x0)), rmask & tmp14 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp0 >= tmp12
        tmp17 = tl.full([1, 1], 624, tl.int64)
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr3 + ((156*r2) + (30576*x1) + ((-468) + x0)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.where(tmp14, tmp15, tmp19)
        tmp21 = tl.where(tmp9, tmp10, tmp20)
        tmp22 = tl.where(tmp4, tmp5, tmp21)
        tmp24 = tmp22 - tmp23
        tmp26 = 0.001
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.sqrt(tmp27)
        tmp29 = tl.full([1, 1], 1, tl.int32)
        tmp30 = tmp29 / tmp28
        tmp31 = 1.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp24 * tmp32
        tmp35 = tmp33 * tmp34
        tmp37 = tmp35 + tmp36
        tmp38 = tl.sigmoid(tmp37)
        tmp39 = tmp37 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp33, rmask & xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tmp43 = 196.0
    tmp44 = tmp41 / tmp43
    tl.store(out_ptr2 + (x3), tmp44, xmask)
