
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__softmax_add_mul_36(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp24 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp4 = tl.full([1, 1], 192, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp7 = tl.full([1, 1], 23, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((23*(triton_helpers.div_floor_integer(11 + (23*(x0 // 8)) + (r2 // 12),  24))) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp5, tmp10, tmp11)
        tmp13 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp14 = tmp13 < tmp4
        tmp15 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp16 = tmp15 < tmp7
        tmp17 = tmp16 & tmp14
        tmp18 = tl.load(in_ptr2 + ((23*((11 + (23*(x0 % 8)) + (r2 % 12)) // 24)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp17, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp14, tmp18, tmp19)
        tmp21 = tmp12 + tmp20
        tmp22 = tmp2 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = triton_helpers.maximum(_tmp24, tmp23)
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = triton_helpers.max2(_tmp24, 1)[:, None]
    _tmp52 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = 0.25
        tmp28 = tmp26 * tmp27
        tmp29 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp30 = tl.full([1, 1], 192, tl.int64)
        tmp31 = tmp29 < tmp30
        tmp32 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp33 = tl.full([1, 1], 23, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp34 & tmp31
        tmp36 = tl.load(in_ptr1 + ((23*(triton_helpers.div_floor_integer(11 + (23*(x0 // 8)) + (r2 // 12),  24))) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp35, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
        tmp38 = tl.where(tmp31, tmp36, tmp37)
        tmp39 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp40 = tmp39 < tmp30
        tmp41 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp42 = tmp41 < tmp33
        tmp43 = tmp42 & tmp40
        tmp44 = tl.load(in_ptr2 + ((23*((11 + (23*(x0 % 8)) + (r2 % 12)) // 24)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
        tmp46 = tl.where(tmp40, tmp44, tmp45)
        tmp47 = tmp38 + tmp46
        tmp48 = tmp28 + tmp47
        tmp49 = tmp48 - tmp24
        tmp50 = tl_math.exp(tmp49)
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
        tmp53 = _tmp52 + tmp51
        _tmp52 = tl.where(rmask, tmp53, _tmp52)
    tmp52 = tl.sum(_tmp52, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp54 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp55 = 0.25
        tmp56 = tmp54 * tmp55
        tmp57 = 11 + (23*(x0 // 8)) + (r2 // 12)
        tmp58 = tl.full([1, 1], 192, tl.int64)
        tmp59 = tmp57 < tmp58
        tmp60 = (11 + (23*(x0 // 8)) + (r2 // 12)) % 24
        tmp61 = tl.full([1, 1], 23, tl.int64)
        tmp62 = tmp60 < tmp61
        tmp63 = tmp62 & tmp59
        tmp64 = tl.load(in_ptr1 + ((23*(triton_helpers.div_floor_integer(11 + (23*(x0 // 8)) + (r2 // 12),  24))) + (184*(x0 % 8)) + (1472*x1) + ((11 + (23*(x0 // 8)) + (r2 // 12)) % 24)), rmask & tmp63, eviction_policy='evict_last', other=0.0)
        tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
        tmp66 = tl.where(tmp59, tmp64, tmp65)
        tmp67 = 11 + (23*(x0 % 8)) + (r2 % 12)
        tmp68 = tmp67 < tmp58
        tmp69 = (11 + (23*(x0 % 8)) + (r2 % 12)) % 24
        tmp70 = tmp69 < tmp61
        tmp71 = tmp70 & tmp68
        tmp72 = tl.load(in_ptr2 + ((23*((11 + (23*(x0 % 8)) + (r2 % 12)) // 24)) + (184*(x0 // 8)) + (1472*x1) + ((11 + (23*(x0 % 8)) + (r2 % 12)) % 24)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp68, tmp72, tmp73)
        tmp75 = tmp66 + tmp74
        tmp76 = tmp56 + tmp75
        tmp77 = tmp76 - tmp24
        tmp78 = tl_math.exp(tmp77)
        tmp79 = tmp78 / tmp52
        tl.store(out_ptr2 + (r2 + (144*x3)), tmp79, rmask)
