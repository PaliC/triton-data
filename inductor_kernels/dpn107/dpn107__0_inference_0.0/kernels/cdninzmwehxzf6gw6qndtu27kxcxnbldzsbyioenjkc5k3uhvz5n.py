
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 7168) % 56
    x1 = (xindex // 128) % 56
    x0 = xindex % 128
    x6 = (xindex // 7168)
    x7 = xindex
    tmp52 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14464) + x0 + (256*x1) + (28672*x6)), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-14336) + x0 + (256*x1) + (28672*x6)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-14208) + x0 + (256*x1) + (28672*x6)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-128) + x0 + (256*x1) + (28672*x6)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + (256*x1) + (28672*x6)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (28672*x6)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14208 + x0 + (256*x1) + (28672*x6)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14336 + x0 + (256*x1) + (28672*x6)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14464 + x0 + (256*x1) + (28672*x6)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp53 = tmp51 - tmp52
    tmp55 = 0.001
    tmp56 = tmp54 + tmp55
    tmp57 = libdevice.sqrt(tmp56)
    tmp58 = tl.full([1], 1, tl.int32)
    tmp59 = tmp58 / tmp57
    tmp60 = 1.0
    tmp61 = tmp59 * tmp60
    tmp62 = tmp53 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tl.full([1], 0, tl.int32)
    tmp68 = triton_helpers.maximum(tmp67, tmp66)
    tmp70 = tmp51 - tmp69
    tmp72 = tmp71 + tmp55
    tmp73 = libdevice.sqrt(tmp72)
    tmp74 = tmp58 / tmp73
    tmp75 = tmp74 * tmp60
    tmp76 = tmp70 * tmp75
    tmp78 = tmp76 * tmp77
    tmp80 = tmp78 + tmp79
    tmp81 = triton_helpers.maximum(tmp67, tmp80)
    tl.store(out_ptr1 + (x7), tmp68, None)
    tl.store(out_ptr2 + (x7), tmp81, None)
