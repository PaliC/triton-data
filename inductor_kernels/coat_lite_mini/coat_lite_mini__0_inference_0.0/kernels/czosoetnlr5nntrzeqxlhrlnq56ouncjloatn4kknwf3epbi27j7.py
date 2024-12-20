
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_mul_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_mul_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 320) % 197
    x4 = xindex % 320
    x5 = (xindex // 320)
    x0 = xindex % 40
    x1 = (xindex // 40) % 8
    x3 = (xindex // 63040)
    x6 = xindex
    tmp38 = tl.load(in_ptr7 + (x0 + (40*x2) + (7880*x1) + (63040*x3)), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (960*x5)), tmp2 & xmask, other=0.0)
    tmp4 = x4
    tmp5 = tmp4 >= tmp1
    tmp6 = tl.full([1], 80, tl.int64)
    tmp7 = tmp4 < tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = tl.load(in_ptr1 + ((80*(((-1) + x2) % 196)) + (15680*x3) + (x0 + (40*x1))), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (40*x1)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp6
    tmp15 = tl.full([1], 200, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = tmp17 & tmp2
    tmp19 = tl.load(in_ptr3 + ((120*(((-1) + x2) % 196)) + (23520*x3) + ((-80) + x0 + (40*x1))), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-80) + x0 + (40*x1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tmp4 >= tmp15
    tmp25 = tl.full([1], 320, tl.int64)
    tmp26 = tmp4 < tmp25
    tmp27 = tmp24 & tmp2
    tmp28 = tl.load(in_ptr5 + ((120*(((-1) + x2) % 196)) + (23520*x3) + ((-200) + x0 + (40*x1))), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + ((-200) + x0 + (40*x1)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp23, tmp32)
    tmp34 = tl.where(tmp7, tmp13, tmp33)
    tmp35 = tmp3 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp39 = 0.15811388300841897
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40 + tmp37
    tl.store(out_ptr1 + (x6), tmp41, xmask)
