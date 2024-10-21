
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp57 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to(1 + ((-1) + x0), [RBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tmp17 & tmp12
    tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp15 >= tmp3
    tmp21 = tmp15 < tmp13
    tmp22 = tmp20 & tmp12
    tmp23 = tl.load(in_ptr4 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp22, other=0.0)
    tmp24 = tl.load(in_ptr5 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp22, other=0.0)
    tmp25 = tl.load(in_ptr6 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp22, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tl.where(tmp17, tmp19, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp11, tmp32)
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tl.full([1], 384, tl.int32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 / tmp42
    tmp44 = tmp34 - tmp43
    tmp45 = tmp44 * tmp44
    tmp46 = tl.broadcast_to(tmp45, [RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp48, 0))
    tmp50 = tmp33 - tmp43
    tmp51 = 384.0
    tmp52 = tmp49 / tmp51
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.rsqrt(tmp54)
    tmp56 = tmp50 * tmp55
    tmp58 = tmp56 * tmp57
    tmp60 = tmp58 + tmp59
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp33, rmask)
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp60, rmask)
