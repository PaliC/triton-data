
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr20': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_floor_mul_rsub_sub_2(in_ptr0, out_ptr20, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 224) % 224
    x0 = xindex % 224
    x2 = (xindex // 50176)
    x3 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 239, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = x0
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 + tmp2
    tmp18 = tmp17 * tmp4
    tmp19 = tmp18 - tmp2
    tmp20 = libdevice.floor(tmp19)
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp21 - tmp9
    tmp23 = triton_helpers.maximum(tmp22, tmp11)
    tmp24 = triton_helpers.minimum(tmp23, tmp13)
    tmp25 = tl.load(in_ptr0 + (tmp24 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp19 - tmp20
    tmp27 = 0.0
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30 + tmp29
    tmp32 = -0.75
    tmp33 = tmp31 * tmp32
    tmp34 = -3.75
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp31
    tmp37 = -6.0
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 * tmp31
    tmp40 = -3.0
    tmp41 = tmp39 - tmp40
    tmp42 = tmp25 * tmp41
    tmp43 = triton_helpers.maximum(tmp21, tmp11)
    tmp44 = triton_helpers.minimum(tmp43, tmp13)
    tmp45 = tl.load(in_ptr0 + (tmp44 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp46 = 1.25
    tmp47 = tmp30 * tmp46
    tmp48 = 2.25
    tmp49 = tmp47 - tmp48
    tmp50 = tmp49 * tmp30
    tmp51 = tmp50 * tmp30
    tmp52 = tmp51 + tmp29
    tmp53 = tmp45 * tmp52
    tmp54 = tmp21 + tmp9
    tmp55 = triton_helpers.maximum(tmp54, tmp11)
    tmp56 = triton_helpers.minimum(tmp55, tmp13)
    tmp57 = tl.load(in_ptr0 + (tmp56 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp58 = tmp29 - tmp30
    tmp59 = tmp58 * tmp46
    tmp60 = tmp59 - tmp48
    tmp61 = tmp60 * tmp58
    tmp62 = tmp61 * tmp58
    tmp63 = tmp62 + tmp29
    tmp64 = tmp57 * tmp63
    tmp65 = tl.full([1], 2, tl.int64)
    tmp66 = tmp21 + tmp65
    tmp67 = triton_helpers.maximum(tmp66, tmp11)
    tmp68 = triton_helpers.minimum(tmp67, tmp13)
    tmp69 = tl.load(in_ptr0 + (tmp68 + (240*tmp14) + (57600*x2)), None, eviction_policy='evict_last')
    tmp70 = 2.0
    tmp71 = tmp70 - tmp30
    tmp72 = tmp71 * tmp32
    tmp73 = tmp72 - tmp34
    tmp74 = tmp73 * tmp71
    tmp75 = tmp74 + tmp37
    tmp76 = tmp75 * tmp71
    tmp77 = tmp76 - tmp40
    tmp78 = tmp69 * tmp77
    tmp79 = triton_helpers.maximum(tmp8, tmp11)
    tmp80 = triton_helpers.minimum(tmp79, tmp13)
    tmp81 = tl.load(in_ptr0 + (tmp24 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp41
    tmp83 = tl.load(in_ptr0 + (tmp44 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp84 = tmp83 * tmp52
    tmp85 = tl.load(in_ptr0 + (tmp56 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp86 = tmp85 * tmp63
    tmp87 = tl.load(in_ptr0 + (tmp68 + (240*tmp80) + (57600*x2)), None, eviction_policy='evict_last')
    tmp88 = tmp87 * tmp77
    tmp89 = tmp8 + tmp9
    tmp90 = triton_helpers.maximum(tmp89, tmp11)
    tmp91 = triton_helpers.minimum(tmp90, tmp13)
    tmp92 = tl.load(in_ptr0 + (tmp24 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp93 = tmp92 * tmp41
    tmp94 = tl.load(in_ptr0 + (tmp44 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp95 = tmp94 * tmp52
    tmp96 = tl.load(in_ptr0 + (tmp56 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp97 = tmp96 * tmp63
    tmp98 = tl.load(in_ptr0 + (tmp68 + (240*tmp91) + (57600*x2)), None, eviction_policy='evict_last')
    tmp99 = tmp98 * tmp77
    tmp100 = tmp8 + tmp65
    tmp101 = triton_helpers.maximum(tmp100, tmp11)
    tmp102 = triton_helpers.minimum(tmp101, tmp13)
    tmp103 = tl.load(in_ptr0 + (tmp24 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp104 = tmp103 * tmp41
    tmp105 = tl.load(in_ptr0 + (tmp44 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp106 = tmp105 * tmp52
    tmp107 = tl.load(in_ptr0 + (tmp56 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp108 = tmp107 * tmp63
    tmp109 = tl.load(in_ptr0 + (tmp68 + (240*tmp102) + (57600*x2)), None, eviction_policy='evict_last')
    tmp110 = tmp109 * tmp77
    tmp111 = tmp42 + tmp53
    tmp112 = tmp111 + tmp64
    tmp113 = tmp112 + tmp78
    tmp114 = tmp6 - tmp7
    tmp115 = triton_helpers.maximum(tmp114, tmp27)
    tmp116 = triton_helpers.minimum(tmp115, tmp29)
    tmp117 = tmp116 + tmp29
    tmp118 = tmp117 * tmp32
    tmp119 = tmp118 - tmp34
    tmp120 = tmp119 * tmp117
    tmp121 = tmp120 + tmp37
    tmp122 = tmp121 * tmp117
    tmp123 = tmp122 - tmp40
    tmp124 = tmp113 * tmp123
    tmp125 = tmp82 + tmp84
    tmp126 = tmp125 + tmp86
    tmp127 = tmp126 + tmp88
    tmp128 = tmp116 * tmp46
    tmp129 = tmp128 - tmp48
    tmp130 = tmp129 * tmp116
    tmp131 = tmp130 * tmp116
    tmp132 = tmp131 + tmp29
    tmp133 = tmp127 * tmp132
    tmp134 = tmp124 + tmp133
    tmp135 = tmp93 + tmp95
    tmp136 = tmp135 + tmp97
    tmp137 = tmp136 + tmp99
    tmp138 = tmp29 - tmp116
    tmp139 = tmp138 * tmp46
    tmp140 = tmp139 - tmp48
    tmp141 = tmp140 * tmp138
    tmp142 = tmp141 * tmp138
    tmp143 = tmp142 + tmp29
    tmp144 = tmp137 * tmp143
    tmp145 = tmp134 + tmp144
    tmp146 = tmp104 + tmp106
    tmp147 = tmp146 + tmp108
    tmp148 = tmp147 + tmp110
    tmp149 = tmp70 - tmp116
    tmp150 = tmp149 * tmp32
    tmp151 = tmp150 - tmp34
    tmp152 = tmp151 * tmp149
    tmp153 = tmp152 + tmp37
    tmp154 = tmp153 * tmp149
    tmp155 = tmp154 - tmp40
    tmp156 = tmp148 * tmp155
    tmp157 = tmp145 + tmp156
    tl.store(out_ptr20 + (x3), tmp157, None)
