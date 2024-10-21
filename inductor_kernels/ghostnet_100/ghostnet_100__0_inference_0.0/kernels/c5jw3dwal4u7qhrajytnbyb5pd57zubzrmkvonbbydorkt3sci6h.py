
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mean_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_mean_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 336, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((336*r2) + (65856*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 672, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + ((336*r2) + (65856*x1) + ((-336) + x0)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.sqrt(tmp14)
        tmp16 = tl.full([1, 1], 1, tl.int32)
        tmp17 = tmp16 / tmp15
        tmp18 = 1.0
        tmp19 = tmp17 * tmp18
        tmp20 = tmp11 * tmp19
        tmp21 = tl.load(in_ptr4 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-336) + x0, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.full([1, 1], 0, tl.int32)
        tmp26 = triton_helpers.maximum(tmp25, tmp24)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp6, tmp26, tmp27)
        tmp29 = tl.where(tmp4, tmp5, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
        tl.store(out_ptr0 + (r2 + (196*x3)), tmp29, rmask & xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tmp33 = 196.0
    tmp34 = tmp31 / tmp33
    tl.store(out_ptr2 + (x3), tmp34, xmask)
