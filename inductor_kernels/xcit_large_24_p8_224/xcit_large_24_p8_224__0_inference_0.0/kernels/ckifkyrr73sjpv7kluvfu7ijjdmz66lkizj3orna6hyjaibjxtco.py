
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp48 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr9 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 785, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (768*((-1) + x0)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr3 + (r2 + (768*(((-1) + x0) % 784)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp12 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tmp9 + tmp14
    tmp16 = tl.load(in_ptr5 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr6 + (r2 + (768*((-1) + x0)) + (602112*x1)), rmask & tmp6, other=0.0)
    tmp18 = tl.load(in_ptr7 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp15 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp5, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tl.full([1], 768, tl.int32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 / tmp33
    tmp35 = tmp25 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tmp24 - tmp34
    tmp42 = 768.0
    tmp43 = tmp40 / tmp42
    tmp44 = 1e-06
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.rsqrt(tmp45)
    tmp47 = tmp41 * tmp46
    tmp49 = tmp47 * tmp48
    tmp51 = tmp49 + tmp50
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp24, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp51, rmask)
