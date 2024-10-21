
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 9424
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = xindex % 62
    x1 = (xindex // 62) % 19
    x2 = (xindex // 1178)
    x5 = xindex % 1178
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7923, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7923*x1)
    tmp4 = tl.full([1, 1], 150528, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((150528*x2) + ((r3 + (128*x0) + (7923*x1)) % 150528)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((192*((r3 + (128*x0) + (7923*x1)) % 784)) + (150528*x2) + (((r3 + (128*x0) + (7923*x1)) // 784) % 192)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + (((r3 + (128*x0) + (7923*x1)) // 784) % 192), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7923*x1)) // 784) % 192), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(xmask, tmp28, 0)
    tmp33 = tl.where(xmask, tmp29, 0)
    tmp34 = tl.where(xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5 + (1184*x2)), tmp38, xmask)
    tl.store(out_ptr1 + (x5 + (1184*x2)), tmp39, xmask)
    tl.store(out_ptr2 + (x5 + (1184*x2)), tmp40, xmask)
