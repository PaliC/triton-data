
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex % 1764
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + (2*x0) + (166*x1) + (6912*x2)), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-83) + (2*x0) + (166*x1) + (6912*x2)), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-82) + (2*x0) + (166*x1) + (6912*x2)), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-1) + (2*x0) + (166*x1) + (6912*x2)), tmp31 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + ((2*x0) + (166*x1) + (6912*x2)), tmp35 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (1 + (2*x0) + (166*x1) + (6912*x2)), tmp39 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x1)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (82 + (2*x0) + (166*x1) + (6912*x2)), tmp47 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (83 + (2*x0) + (166*x1) + (6912*x2)), tmp51 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (84 + (2*x0) + (166*x1) + (6912*x2)), tmp55 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x4 + (1792*x2)), tmp57, xmask)
