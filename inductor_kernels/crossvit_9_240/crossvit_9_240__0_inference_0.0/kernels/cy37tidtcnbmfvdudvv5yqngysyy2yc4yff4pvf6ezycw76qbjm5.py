
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr3': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 13, 'num_reduction': 8, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_gelu_native_layer_norm_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr7, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1 + (50432*x0)), None)
    tmp2 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp35 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr7 + (r1), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr8 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp18 >= tmp18
    tmp20 = tl.full([1], 1, tl.int64)
    tmp21 = tmp18 < tmp20
    tmp22 = tl.load(in_ptr3 + (r1 + (256*x0)), tmp21, other=0.0)
    tmp23 = tmp18 >= tmp20
    tmp24 = tl.full([1], 197, tl.int64)
    tmp25 = tmp18 < tmp24
    tmp26 = tl.load(in_ptr0 + (256 + r1 + (256*(-1)) + (50432*x0)), tmp23, other=0.0)
    tmp27 = tl.load(in_ptr1 + (256 + r1 + (256*(-1)) + (50432*x0)), tmp23, other=0.0)
    tmp28 = tl.load(in_ptr2 + (tl.broadcast_to(r1, [RBLOCK])), tmp23, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp21, tmp22, tmp32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tmp4 - tmp12
    tmp39 = 256.0
    tmp40 = tmp17 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = 0.5
    tmp50 = tmp48 * tmp49
    tmp51 = 0.7071067811865476
    tmp52 = tmp48 * tmp51
    tmp53 = libdevice.erf(tmp52)
    tmp54 = 1.0
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 * tmp55
    tmp57 = tl.broadcast_to(tmp37, [RBLOCK])
    tmp59 = tl.broadcast_to(tmp57, [RBLOCK])
    tmp61 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp62 = tmp61 / tmp11
    tmp63 = tmp57 - tmp62
    tmp64 = tmp63 * tmp63
    tmp65 = tl.broadcast_to(tmp64, [RBLOCK])
    tmp67 = triton_helpers.promote_to_tensor(tl.sum(tmp65, 0))
    tmp68 = tmp37 - tmp62
    tmp69 = tmp67 / tmp39
    tmp70 = tmp69 + tmp41
    tmp71 = libdevice.rsqrt(tmp70)
    tmp72 = tmp68 * tmp71
    tmp74 = tmp72 * tmp73
    tmp76 = tmp74 + tmp75
    tmp77 = tmp76 * tmp49
    tmp78 = tmp76 * tmp51
    tmp79 = libdevice.erf(tmp78)
    tmp80 = tmp79 + tmp54
    tmp81 = tmp77 * tmp80
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp37, None)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp56, None)
    tl.store(out_ptr7 + (r1 + (256*x0)), tmp81, None)
