
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_layer_norm_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_cat_native_layer_norm_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    x0 = xindex % 50
    r2 = rindex
    x1 = (xindex // 50)
    x3 = xindex
    tmp17 = tl.load(in_out_ptr0 + (r2 + (512*x3)), None)
    tmp18 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x0) + (25600*x1)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 50, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (512*(((-1) + x0) % 49)) + (25088*x1)), tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr0 + (512 + r2 + (512*(((-1) + x0) % 49)) + (25600*x1)), tmp6, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp21 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp20 - tmp28
    tmp35 = 512.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp20, None)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp44, None)
