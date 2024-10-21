
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x0 = xindex % 432
    x5 = (xindex // 9072)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (864*x1) + (36288*x5)), tmp5 & xmask, other=float("-inf"))
    tmp7 = 1 + (2*x1)
    tmp8 = tmp7 < tmp1
    tmp9 = tmp2 & tmp8
    tmp10 = tl.load(in_ptr0 + (432 + x0 + (864*x1) + (36288*x5)), tmp9 & xmask, other=float("-inf"))
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 2 + (2*x1)
    tmp13 = tmp12 < tmp1
    tmp14 = tmp2 & tmp13
    tmp15 = tl.load(in_ptr0 + (864 + x0 + (864*x1) + (36288*x5)), tmp14 & xmask, other=float("-inf"))
    tmp16 = triton_helpers.maximum(tmp15, tmp11)
    tmp17 = 1 + (2*x2)
    tmp18 = tmp17 < tmp1
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr0 + (18144 + x0 + (864*x1) + (36288*x5)), tmp19 & xmask, other=float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp16)
    tmp22 = tmp18 & tmp8
    tmp23 = tl.load(in_ptr0 + (18576 + x0 + (864*x1) + (36288*x5)), tmp22 & xmask, other=float("-inf"))
    tmp24 = triton_helpers.maximum(tmp23, tmp21)
    tmp25 = tmp18 & tmp13
    tmp26 = tl.load(in_ptr0 + (19008 + x0 + (864*x1) + (36288*x5)), tmp25 & xmask, other=float("-inf"))
    tmp27 = triton_helpers.maximum(tmp26, tmp24)
    tmp28 = 2 + (2*x2)
    tmp29 = tmp28 < tmp1
    tmp30 = tmp29 & tmp4
    tmp31 = tl.load(in_ptr0 + (36288 + x0 + (864*x1) + (36288*x5)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp27)
    tmp33 = tmp29 & tmp8
    tmp34 = tl.load(in_ptr0 + (36720 + x0 + (864*x1) + (36288*x5)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp13
    tmp37 = tl.load(in_ptr0 + (37152 + x0 + (864*x1) + (36288*x5)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tl.store(out_ptr0 + (x6), tmp38, xmask)
    tl.store(out_ptr1 + (x6), tmp38, xmask)
