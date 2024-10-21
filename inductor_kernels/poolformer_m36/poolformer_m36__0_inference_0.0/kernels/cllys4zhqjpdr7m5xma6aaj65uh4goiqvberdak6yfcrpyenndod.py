
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_29(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + (((r3 + (128*x1) + (7527*x0)) // 196) % 384), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = 0.0
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = 1.0
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
    tmp23 = tl.where(tmp2, tmp21, tmp22)
    tmp24 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp24, 0)
    tmp29 = tl.where(xmask, tmp25, 0)
    tmp30 = tl.where(xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x4), tmp34, xmask)
    tl.store(out_ptr1 + (x4), tmp35, xmask)
    tl.store(out_ptr2 + (x4), tmp36, xmask)
