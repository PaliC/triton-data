
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_convolution_mul_sub_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 56) % 56
    y0 = yindex % 56
    x3 = xindex
    y6 = yindex
    y2 = (yindex // 3136)
    y4 = yindex % 3136
    tmp54 = tl.load(in_ptr1 + (x3 + (96*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x3 + (96*y6)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5472) + x3 + (96*y6)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5376) + x3 + (96*y6)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5280) + x3 + (96*y6)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-96) + x3 + (96*y6)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x3 + (96*y6)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (96 + x3 + (96*y6)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5280 + x3 + (96*y6)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5376 + x3 + (96*y6)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5472 + x3 + (96*y6)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = (((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))) + (((56) * ((56) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (56)))*((56) * ((56) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + y0)) + ((-1) + y0) * (((-1) + y0) > (0)))*((56) * ((56) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (56)))) + ((-1)*((0) * ((0) >= ((-1) + y1)) + ((-1) + y1) * (((-1) + y1) > (0)))*((56) * ((56) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (56))))
    tmp53 = tmp51 / tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp53 - tmp57
    tmp60 = tmp58 * tmp59
    tmp61 = tmp56 + tmp60
    tl.store(out_ptr1 + (y4 + (3136*x3) + (301056*y2)), tmp61, xmask & ymask)