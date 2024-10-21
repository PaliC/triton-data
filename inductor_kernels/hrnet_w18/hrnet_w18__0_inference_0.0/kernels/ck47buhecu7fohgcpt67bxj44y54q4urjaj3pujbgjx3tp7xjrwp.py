
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 144
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y0 = yindex % 18
    y1 = (yindex // 18)
    x4 = xindex
    y5 = yindex
    tmp10 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr13 + (y0), ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr14 + (y0), ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (y0 + (18*tmp8) + (504*tmp4) + (14112*y1)), xmask & ymask)
    tmp11 = tmp9 - tmp10
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.sqrt(tmp14)
    tmp16 = tl.full([1, 1], 1, tl.int32)
    tmp17 = tmp16 / tmp15
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp11 * tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = 0.25
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tmp6 * tmp25
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr5 + (y0 + (18*tmp29) + (252*tmp27) + (3528*y1)), xmask & ymask)
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp13
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp16 / tmp35
    tmp37 = tmp36 * tmp18
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = 0.125
    tmp44 = tmp1 * tmp43
    tmp45 = tmp44.to(tl.int32)
    tmp46 = tmp6 * tmp43
    tmp47 = tmp46.to(tl.int32)
    tmp48 = tl.load(in_ptr10 + (y0 + (18*tmp47) + (126*tmp45) + (882*y1)), xmask & ymask)
    tmp50 = tmp48 - tmp49
    tmp52 = tmp51 + tmp13
    tmp53 = libdevice.sqrt(tmp52)
    tmp54 = tmp16 / tmp53
    tmp55 = tmp54 * tmp18
    tmp56 = tmp50 * tmp55
    tmp58 = tmp56 * tmp57
    tmp60 = tmp58 + tmp59
    tmp62 = tmp61 + tmp24
    tmp63 = tmp62 + tmp42
    tmp64 = tmp63 + tmp60
    tmp65 = tl.full([1, 1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (18*x4) + (56448*y1)), tmp66, xmask & ymask)
